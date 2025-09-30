import logging
import json
import toml
import signal
import sys
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from docker_setup import DockerEnvironmentManager, ContainerOperator, AgentManager
from pytest_output_parse import TestStatus
from patch_analyzer import PatchAnalyzer
from tqdm import tqdm

@dataclass
class AgentConfig:
    """Agent configuration class"""
    name: str
    repo_url: str
    branch: str = "main"
    install_command: str = ""
    model: str = ""
    provider: str = ""
    extra_env: Dict[str, str] = None
    
    def __post_init__(self):
        if self.extra_env is None:
            self.extra_env = {}

class AgentEvaluator:
    """Agent evaluator"""
    
    def __init__(self, config_path: str = "config.toml"):
        # Set base path first
        self.base_path = Path(__file__).parent
        self.docker_manager = DockerEnvironmentManager()

        # Read configuration file (supports relative paths)
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = self.base_path / cfg_path
        self._raw_config = toml.load(cfg_path)

        # Map common configurations to attributes for easy access
        logging_cfg = self._raw_config.get("logging", {})
        self.log_level = logging_cfg.get("level", "INFO")
        self.log_format = logging_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.log_file = logging_cfg.get("log_file", "logs/evaluation.log")

        paths_cfg = self._raw_config.get("paths", {})
        analysis_path = paths_cfg.get("analysis_file", "logs/analysis_results_part.json")
        self.analysis_file = Path(analysis_path) if isinstance(analysis_path, str) else Path("analysis_results_part.json")

        eval_cfg = self._raw_config.get("evaluation", {})
        self.default_timeout = eval_cfg.get("default_timeout", 1800)
        self.max_instances_per_repo = eval_cfg.get("max_instances_per_repo", 100)

        agent_cfg = self._raw_config.get("agents", {})
        self.agent_config_file = agent_cfg.get("config_file", "agent_configs.json")
        self.agentless_file = agent_cfg.get("agentless_file", "output_0_processed.jsonl")

        # Track current active containers for signal handler use
        self.active_containers: List[Any] = []
         
        self.cleanup_in_progress = False
        
        # Configure logging
        self._setup_logging()
        
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(__name__)
        
        # Load agent configurations
        self.agents = self._load_agent_configs()

        # Initialize patch analyzer
        self.patch_analyzer = PatchAnalyzer()

        # Agentless patch storage
        self.agentless_patches: List[Dict[str, Any]] = []

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format=self.log_format,
            handlers=[
                logging.FileHandler(self.base_path / self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle termination signal"""
        if self.cleanup_in_progress:
            return
            
        self.cleanup_in_progress = True
        self.logger.info(f"\nReceived signal {signum}, cleaning up containers...")
        
        for container in list(self.active_containers):
            if container:
                try:
                    self.docker_manager.cleanup_container(container, force_remove=True)
                    if container in self.active_containers:
                        self.active_containers.remove(container)
                except Exception as e:
                    self.logger.error(f"Error cleaning up container: {e}")
        
        sys.exit(0)

    def _load_agent_configs(self) -> List[AgentConfig]:
        """Load agent configurations"""
        config_file = self.base_path / self.agent_config_file
        if not config_file.exists():
            # Create default configuration file
            default_configs = [
                {
                    "name": "trae-agent",
                    "repo_url": "https://github.com/bytedance/trae-agent.git",
                    "branch": "main",
                    "install_command": "uv sync --all-extras",
                    "model": "deepseek-chat",
                    "provider": "deepseek",
                },
                {
                    "name": "agentless",
                    "repo_url": "agentless",
                    "branch": "main",
                    "install_command": "uv venv -p 3.11 && uv pip install -r requirements.txt",
                    "model": "deepseek-chat",
                    "provider": "deepseek"
                }
            ]
            
            with config_file.open("w", encoding="utf-8") as f:
                json.dump(default_configs, f, indent=2, ensure_ascii=False)
            
        with config_file.open("r", encoding="utf-8") as f:
            configs_data = json.load(f)
            
        return [AgentConfig(**config) for config in configs_data]

    def _load_specs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load and group specs by repository"""
        with self.analysis_file.open("r", encoding="utf-8") as f:
            specs = json.load(f)

        specs_by_repo = defaultdict(list)
        for spec in specs:
            repo = spec["repo"]
            specs_by_repo[repo].append(spec)
        
        return specs_by_repo

    def _save_evaluation_results(self, results: List[Dict[str, Any]]):
        """Save evaluation results (append mode)"""
        results_file = self.base_path / "data" / "processed_evaluation_results_agentless.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing content first
        existing_results = []
        if results_file.exists():
            with results_file.open("r", encoding="utf-8") as f:
                try:
                    existing_results = json.load(f)
                except Exception:
                    existing_results = []
        
        # Merge and deduplicate (deduplicate by instance_id)
        all_results = existing_results + results
        seen = set()
        deduped_results = []
        for r in all_results:
            iid = r.get("instance_id")
            if iid and iid not in seen:
                deduped_results.append(r)
                seen.add(iid)
            elif not iid:
                deduped_results.append(r)  # Keep those without instance_id
        
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(deduped_results, f, indent=2, ensure_ascii=False)

    def _clean_ansi_codes(self, text: str) -> str:
        """Clean ANSI escape codes"""
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)

    def _parse_agent_log(self, log: str) -> int:
        """
        Parse agent log to extract useful information
        """
        # Clean ANSI escape codes
        clean_log = self._clean_ansi_codes(log)

        execution_summary_start = clean_log.find("Execution Summary")
        summary_section = clean_log[execution_summary_start:]

        # Extract Total Tokens
        total_tokens = None
        for line in summary_section.split('\n'):
            line = line.strip()
            if line.startswith("│ Total Tokens"):
                # Extract number
                match = re.search(r'│ Total Tokens\s*│\s*(\d+)', line)
                if match:
                    total_tokens = int(match.group(1))
                    break

        return total_tokens

    def _evaluate_agent_on_spec(self, agent_config: AgentConfig, container, spec: Dict[str, Any], repo_name: str) -> Dict[str, Any]:
        """Evaluate single agent performance on specific spec"""
        self.logger.info(f"Starting evaluation of {agent_config.name} on {spec['instance_id']}")
        
        operator = ContainerOperator(spec["repo"], container)
        agent_manager = AgentManager(container, agent_config)
        
        try:
            # Set up agent environment
            if agent_config.name == "trae-agent":
                agent_manager.setup_agent()
            
            # Prepare problem environment
            operator.checkout_commit(spec["base_commit"], use_docker=True)
            
            # Run agent
            if agent_config.name == "trae-agent":
                agent_success, agent_output = agent_manager.run_agent_on_problem(
                    spec["problem_statement"],
                    spec["instance_id"],
                    repo_name
                )
            elif agent_config.name == "agentless":
                instance_id = spec["instance_id"]
                agentless_patch = next((p for p in self.agentless_patches if p.get("instance_id") == instance_id), None)
                if agentless_patch:
                    patch_file = self.base_path / "swap" / repo_name / "patch.diff"
                    agent_success = patch_file.write_text(agentless_patch["model_patch"], encoding="utf-8")
                else:
                    agent_success = False
                    agent_output = "No agentless patch found"
            
            # Evaluate results
            if agent_success:
                operator.checkout_commit(spec["base_commit"], exclude_file=["patch.diff"], use_docker=True)
                patch_application = self._apply_patches(operator, repo_name)
                operator.apply_patches(spec["test_patch"])
                
                # Parse test list directly from spec
                f2p_tests, p2p_tests = [], []
                if "FAIL_TO_PASS" in spec:
                    f2p_tests.extend(spec["FAIL_TO_PASS"].split(", "))
                if "PASS_TO_PASS" in spec:
                    p2p_tests.extend(spec["PASS_TO_PASS"].split(", "))
                
                # Run specified tests
                if f2p_tests:
                    f2p_passed, f2p_logs = operator.run_tests_in_container(
                        repo_name, f2p_tests, [TestStatus.PASSED], False
                    )

                operator.checkout_commit(spec["base_commit"], exclude_file=["patch.diff"], use_docker=True)
                patch_application = self._apply_patches(operator, repo_name)
                operator.apply_patches(spec["test_patch"])

                if p2p_tests:
                    p2p_passed, p2p_logs = operator.run_tests_in_container(
                        repo_name, p2p_tests, [TestStatus.PASSED]
                    )
                
                # Check if all expected tests pass
                success_f2p = all(test in f2p_passed for test in f2p_tests)
                success_p2p = all(test in p2p_passed for test in p2p_tests)
                success = success_f2p and success_p2p
                
                evaluation_result = {
                    "agent": agent_config.name,
                    "model": agent_config.model,
                    "instance_id": spec["instance_id"],
                    "success_f2p": success_f2p,
                    "success_p2p": success_p2p,
                    "success": success,
                    "passed_f2p_tests": list(f2p_passed),
                    "passed_p2p_tests": list(p2p_passed),
                    "expected_f2p_tests": f2p_tests,
                    "expected_p2p_tests": p2p_tests,
                    "total_tokens": self._parse_agent_log(agent_output) if agent_config.name == "trae-agent" else None,
                    "patch_application": patch_application,
                }
            else:
                evaluation_result = {
                    "agent": agent_config.name,
                    "model": agent_config.model,
                    "instance_id": spec["instance_id"],
                    "success": False,
                    "error": "Agent failed to generate valid patches"
                }
            
            self.logger.info(f"Evaluation completed: {agent_config.name} on {spec['instance_id']}, success: {evaluation_result['success']}")
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating {agent_config.name} on {spec['instance_id']}: {str(e)}")
            return {
                "agent": agent_config.name,
                "model": agent_config.model,
                "instance_id": spec["instance_id"],
                "success": False,
                "error": str(e)
            }
        
    def _apply_patches(self, operator: ContainerOperator, repo_name: str):
        """Apply patches in container - fully use unified patch analyzer"""
        patch_path = self.base_path / "swap" / repo_name / "patch.diff"
        
        try:
            # Directly use patch analyzer's file application function
            workdir = f"/workdir/swap/{repo_name}"
            result = self.patch_analyzer.apply_patch_file_to_container(
                patch_path, operator.docker_executor, workdir, include_test=False, include_source=True
            )
            
            # Record application results and statistics
            self.logger.info(f"Patch application result: Total {result['total_files_num']} patches, successfully applied {result['applied_files_num']}")
            self.logger.info(f"Applied files: {result['applied_files']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying patch: {e}")
            return None
            
    def evaluate(self, agent_names: Optional[List[str]] = None, max_instances: int = 10):
        """Main evaluation method"""
        specs_by_repo = self._load_specs()
        
        # Read processed instances
        results_file = self.base_path / "data" / "processed_evaluation_results_agentless.json"
        processed_instance_ids = set()
        if results_file.exists():
            with results_file.open("r", encoding="utf-8") as f:
                try:
                    prev_results = json.load(f)
                    processed_instance_ids = {r.get("instance_id") for r in prev_results if "instance_id" in r}
                except Exception:
                    processed_instance_ids = set()
        
        # Filter agents to evaluate
        agents_to_evaluate = self.agents
        if agent_names:
            agents_to_evaluate = [agent for agent in self.agents if agent.name in agent_names]

        if "agentless" in agent_names:
            self.agentless_patches = []
            agentless_file = self.base_path / self.agentless_file
            if agentless_file.exists():
                with agentless_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            spec = json.loads(line)
                            if spec.get("model_patch"):
                                self.agentless_patches.append(spec)
                        except Exception as e:
                            self.logger.error(f"Error parsing agentless file line: {e}")
        
        all_results = []
        
        # Use tqdm to show repo progress
        for repo, repo_specs in tqdm(list(specs_by_repo.items()), desc="Repo progress", unit="repo"):
            # Limit instances per repo
            specs_to_test = repo_specs[:max_instances]
            
            # Use tqdm to show instance progress
            for spec in tqdm(specs_to_test, desc=f"{repo.split('/')[-1]} instances", unit="spec", leave=False):
                if not spec.get("FAIL_TO_PASS"):
                    continue  # Only evaluate instances with clear test targets

                if spec.get("instance_id") in processed_instance_ids:
                    continue

                repo_name = repo.split('/')[-1]
                container = None
                
                try:
                    # Create container
                    container = self.docker_manager.setup_container_and_environment(
                        repo, spec["instance_id"].split("-")[-1]
                    )
                    # Record active containers for cleanup on signal
                    self.active_containers.append(container)
                     
                    # Evaluate each agent
                    for agent_config in agents_to_evaluate:
                        result = self._evaluate_agent_on_spec(
                            agent_config, container, spec, repo_name
                        )
                        all_results.append(result)
                        
                        # Save results immediately
                        self._save_evaluation_results(all_results)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {spec['instance_id']}: {str(e)}")
                finally:
                    if container:
                        try:
                            self.docker_manager.cleanup_container(container, force_remove=True)
                        finally:
                            if container in self.active_containers:
                                self.active_containers.remove(container)

        self.logger.info(f"Evaluation completed, evaluated {len(all_results)} instances")
        return all_results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Evaluator")
    parser.add_argument("--agents", default="trae-agent", nargs="+", help="List of agent names to evaluate")
    parser.add_argument("--max-instances", type=int, default=100, help="Maximum instances per repo")
    args = parser.parse_args()
    
    evaluator = AgentEvaluator()
    evaluator.evaluate(agent_names=args.agents, max_instances=args.max_instances)

if __name__ == "__main__":
    main()