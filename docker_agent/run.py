import logging
import json
import signal
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

from agent_config import AgentConfig
from agent_executor import AgentExecutor, AgentTaskType
from docker_setup import DockerEnvironmentManager, ContainerOperator
from locate_test import CodeChange, CodeChangeAnalyzer, PytestFilter
from pytest_output_parse import TestStatus

class DockerAgentRunner:
    """Docker Agent runner class"""
    
    def __init__(self, config_path: str = "config.toml", test_only: bool = False):
        self.config = AgentConfig(config_path)
        self.docker_executor = AgentExecutor(self.config, use_docker=True)
        self.local_executor = AgentExecutor(self.config, use_docker=False)
        self.active_containers = []
        self.cleanup_in_progress = False
        self.docker_manager = DockerEnvironmentManager()
        self.base_path = Path(__file__).parent
        self.test_only = test_only
        
        # Configure logging
        self._setup_logging()
        
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.FileHandler(self.config.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle termination signal"""
        if self.cleanup_in_progress:
            self.logger.info("Cleanup already in progress, ignoring duplicate signal")
            return
            
        self.cleanup_in_progress = True
        self.logger.info(f"\nReceived signal {signum}, cleaning up containers...")
        
        for container in self.active_containers[:]:
            if container:
                try:
                    try:
                        response = input(f"\nDo you want to delete container {container.name}? (y/N): ").strip().lower()
                        force_remove = response in ['y', 'yes']
                    except (EOFError, KeyboardInterrupt):
                        force_remove = False
                        self.logger.info("User interrupted input, defaulting to keep container")
                    
                    self.docker_manager.cleanup_container(container, force_remove=force_remove)
                    self.active_containers.remove(container)
                except Exception as e:
                    self.logger.error(f"Error cleaning up container {container.name}: {e}")
        
        self.cleanup_in_progress = False
        sys.exit(0)

    def _load_specs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load and group specs by repository"""
        with self.config.analysis_file.open("r", encoding="utf-8") as f:
            specs = json.load(f)

        specs_by_repo = defaultdict(list)
        for spec in specs:
            repo = spec["repo"]
            specs_by_repo[repo].append(spec)
        
        return specs_by_repo

    def _save_specs(self, specs_by_repo: Dict[str, List[Dict[str, Any]]]):
        """Save specs to file"""
        updated_specs = []
        for all_repo_specs in specs_by_repo.values():
            updated_specs.extend(all_repo_specs)
        
        with self.config.analysis_file.open("w", encoding="utf-8") as f:
            json.dump(updated_specs, f, indent=2, ensure_ascii=False)

    def _setup_repo_environment(self, container, repo: str, repo_name: str, spec: Dict[str, Any]):
        """Set up repository environment"""
        self.logger.info(f"Second stage: Configure environment for repository {repo}")

        self._restore_setup_files(repo, repo_name)
        
        self.docker_executor.call_trae_agent(
            repo_name,
            spec["instance_id"], AgentTaskType.ENV_SETUP, [file for file in spec["test_files"] if file.endswith(".py")], spec["created_at"], container
        )

    def _prepare_setup_files(self, repo: str, repo_name: str, spec: Dict[str, Any]):
        # Check if configuration file list for this repository already exists
        swap_dir = self.base_path / "swap"
        setup_files_json = swap_dir / "setup_files_list.json"
        operator = ContainerOperator(repo=repo)
        
        if setup_files_json.exists():
            try:
                with setup_files_json.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                if repo.replace("/", "_") in existing_data:
                    operator.checkout_commit(spec["base_commit"], use_docker=False)
                    self.logger.info(f"Configuration file list for repository {repo} already exists, skipping first stage")
                    return
            except Exception as e:
                self.logger.warning(f"Error reading existing configuration file list: {e}")
        
        operator.repo_clone(use_docker=False)

        operator.checkout_commit(spec["base_commit"], use_docker=False)

        self.logger.info(f"First stage: List environment configuration files for repository {repo}")
        self.local_executor.call_trae_agent( 
            repo_name, 
            spec['instance_id'], AgentTaskType.FILE_LIST
        )
        self._transfer_and_merge_setup_files(repo, repo_name)

    def _transfer_and_merge_setup_files(self, repo: str, repo_name: str):
        """Transfer generated JSON files to swap directory and merge by repository"""
        try:
            # Define file paths
            base_dir = self.base_path / "swap" / repo_name
            swap_dir = self.base_path / "swap"
            
            # Define files to process
            files_to_process = [
                "recommended_python_version.json",
                "setup_files_list.json"
            ]
            
            for filename in files_to_process:
                if filename == "recommended_python_version.json":
                    source_file = base_dir / filename
                    target_file = swap_dir / filename
                    if source_file.exists():
                        with source_file.open("r", encoding="utf-8") as f:
                            new_data = f.read().strip()
                    else:
                        self.logger.warning(f"Source file does not exist: {source_file}")
                        continue

                    # Merge into json
                    merged_data = {}
                    if target_file.exists():
                        with target_file.open("r", encoding="utf-8") as f:
                            merged_data = json.load(f)
                    merged_data[repo.replace("/", "_")] = new_data
                    with target_file.open("w", encoding="utf-8") as f:
                        json.dump(merged_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Transferred and merged {filename} to {target_file}")
                    source_file.unlink()
                else:
                    source_file = base_dir / filename
                    target_file = swap_dir / filename
                    if source_file.exists():
                        with source_file.open("r", encoding="utf-8") as f:
                            new_data = json.load(f)
                        merged_data = {}
                        if target_file.exists():
                            with target_file.open("r", encoding="utf-8") as f:
                                merged_data = json.load(f)
                        merged_data[repo.replace("/", "_")] = new_data
                        with target_file.open("w", encoding="utf-8") as f:
                            json.dump(merged_data, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"Transferred and merged {filename} to {target_file}")
                        source_file.unlink()
                    else:
                        self.logger.warning(f"Source file does not exist: {source_file}")
                    
        except Exception as e:
            self.logger.error(f"Error transferring and merging setup files: {str(e)}")

    def _restore_setup_files(self, repo: str, repo_name: str):
        """Restore configuration files from swap directory to corresponding repository directory"""
        try:
            # Define file paths
            base_dir = self.base_path / "swap" / repo_name
            swap_dir = self.base_path / "swap"
            
            # Ensure target directory exists
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Define files to restore
            files_to_restore = [
                # Only output json
                "recommended_python_version.json",
                "setup_files_list.json"
            ]
            
            for filename in files_to_restore:
                source_file = swap_dir / filename
                target_file = base_dir / filename
                
                if source_file.exists():
                    with source_file.open("r", encoding="utf-8") as f:
                        merged_data = json.load(f)
                    if repo.replace("/", "_") in merged_data:
                        repo_data = merged_data[repo.replace("/", "_")]
                        with target_file.open("w", encoding="utf-8") as f:
                            json.dump(repo_data, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"Restored {filename} to {target_file}")
                    else:
                        self.logger.warning(f"Data for repository {repo} not found in {filename}")
                else:
                    self.logger.warning(f"Merged file does not exist: {source_file}")
                    
        except Exception as e:
            self.logger.error(f"Error restoring setup files: {str(e)}")

    def _save_test_logs(self, repo_name: str, pre_logs: str, post_logs: str):
        """Save test logs to logs/test_logs.json (simplified version)"""
        logs_file = self.base_path / "logs" / "test_logs.json"
        logs_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            existing_logs = {}
            if logs_file.exists():
                with logs_file.open("r", encoding="utf-8") as f:
                    existing_logs = json.load(f)

            existing_logs[repo_name] = {
                "pre_logs": pre_logs,
                "post_logs": post_logs
            }

            with logs_file.open("w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save test logs: {e}")

    def _reset_and_apply(self, operator: ContainerOperator, repo_name: str, base_commit: str, patches: List[List[Dict[str, Any]]]):
        operator.checkout_commit(base_commit, use_docker=True)
        for p in patches or []:
            if p:
                operator.apply_patches(p)

    def _run_tests(self, operator: ContainerOperator, repo_name: str, test_filter = Optional[List[Dict[str, CodeChange]]], expected_statuses: List[TestStatus] = [TestStatus.PASSED], use_xdist = True):
        if test_filter is None:
            passed, logs = operator.run_tests_in_container(repo_name, expected_statuses=expected_statuses, use_xdist=use_xdist)
        else:
            passed, logs = operator.run_tests_in_container(repo_name, test_filter, expected_statuses, use_xdist=use_xdist)
        return set(passed), logs

    def _process_spec(self, container, spec: Dict[str, Any], repo_name: str):
        """Process single spec (decoupled, using helper)"""

        operator = ContainerOperator(repo_name, container)

        # Get test code before
        self._reset_and_apply(operator, repo_name, spec["base_commit"], [])
        test_code_before = self._get_test_code(spec, repo_name)

        # Apply test patch and get test code after
        self._reset_and_apply(operator, repo_name, spec["base_commit"], [spec.get("test_patch")])
        test_code_after = self._get_test_code(spec, repo_name)

        # Calculate changed test functions
        test_func = self._get_test_func(test_code_before, test_code_after)
        if all(not changes for changes_dict in test_func for changes in changes_dict.values()):
            self.logger.info(f"Skipping test for spec {spec['instance_id']}")
            spec["processed"] = True
            return
        
        # f2p: Before patch (only apply test_patch) for selected test functions, expect fail or error
        self._reset_and_apply(operator, repo_name, spec["base_commit"], [spec.get("test_patch")])
        f2p_failed, f2p_pre_logs = self._run_tests(operator, repo_name, test_func, [TestStatus.FAILED, TestStatus.ERROR], False)

        # f2p: After applying patch for selected test functions, expect pass
        self._reset_and_apply(operator, repo_name, spec["base_commit"], [spec.get("test_patch"), spec.get("patch")])
        f2p_passed, f2p_post_logs = self._run_tests(operator, repo_name, test_func, [TestStatus.PASSED], False)

        # p2p: Tests that pass before and after patch (first run with only test_patch applied)
        self._reset_and_apply(operator, repo_name, spec["base_commit"], [spec.get("test_patch")])
        p2p_pre_passed, p2p_pre_logs = self._run_tests(operator, repo_name, None, [TestStatus.PASSED])

        # p2p: Run again after applying patch (only care about passed set)
        self._reset_and_apply(operator, repo_name, spec["base_commit"], [spec.get("test_patch"), spec.get("patch")])
        p2p_post_passed, p2p_post_logs = self._run_tests(operator, repo_name, None, [TestStatus.PASSED])

        # Log recording
        self.logger.info(f"Test files that failed before patch: {sorted(f2p_failed)}")
        self.logger.info(f"Test files that passed before patch: {sorted(p2p_pre_passed)[:5]}")
        self.logger.info(f"Test files that passed after patch: {sorted(f2p_passed)}")
        self.logger.info(f"Test files that still passed after patch: {sorted(p2p_post_passed)[:5]}")

        # Save test logs
        self._save_test_logs(repo_name, p2p_pre_logs, p2p_post_logs)

        # Calculate results
        fail_to_pass = f2p_failed & f2p_passed
        pass_to_pass = p2p_pre_passed & p2p_post_passed

        spec["FAIL_TO_PASS"] = ", ".join(sorted(fail_to_pass)) if fail_to_pass else None
        spec["PASS_TO_PASS"] = ", ".join(sorted(pass_to_pass)) if pass_to_pass else None
        spec["processed"] = True

        self.logger.info("=== Test Results Summary ===")
        self.logger.info(f"Tests that only passed after patch: {spec['FAIL_TO_PASS']}")
        self.logger.info(f"Tests that passed both before and after patch: {spec['PASS_TO_PASS']}")
    
    def _get_test_code(self, spec: Dict[str, Any], repo_name: str):
        test_py = []
        for f in spec["test_files"]:
            if f.endswith(".py"):
                try:
                    test_py.append(Path(self.base_path / "swap" / repo_name / f).read_text(encoding="utf-8", errors='replace'))
                except FileNotFoundError:
                    test_py.append("")

        file_names = [f for f in spec["test_files"] if f.endswith(".py")]
        return [{name: text} for name, text in zip(file_names, test_py)]
    
    def _get_test_func(self, code_before: List[Dict[str, Any]], code_after: List[Dict[str, Any]]) -> List[Dict[str, CodeChange]]:
        analyzer = CodeChangeAnalyzer()
        pytest_filter = PytestFilter()
        result = []
        for before, after in zip(code_before, code_after):
            file_name = list(before.keys())[0]
            before_code = before[file_name]
            after_code = after[file_name]
            changes = analyzer.analyze_changes(before_code, after_code)
            pytest_changes = pytest_filter.filter_pytest_changes(changes)
            result.append({file_name: pytest_changes})
        return result

    def run(self):
        """Main run method"""
        specs_by_repo = self._load_specs()

        for repo, repo_specs in list(specs_by_repo.items()):
            for spec in repo_specs[:self.config.max_specs_per_repo]:
                if not self.test_only:
                    if spec.get("processed", False):
                        self.logger.info(f"Skipping processed spec: {spec['instance_id']}")
                        continue
                else:
                    if spec.get("FAIL_TO_PASS", None) is None:
                        continue
                    if spec.get("PASS_TO_PASS", None) is not None:
                        continue

                container = None
                repo_name = repo.split('/')[-1]
                
                try:
                    if not self.test_only:
                        self._prepare_setup_files(repo, repo_name, spec)
                        container = self.docker_manager.setup_container_and_environment(repo, spec["instance_id"].split("-")[-1])
                        try:
                            self._setup_repo_environment(container, repo, repo_name, spec)
                            
                            # Save image
                            try:
                                self.docker_manager.cache_manager.save_container_as_image(container)
                                self.logger.info(f"Saved configured image for repository {repo.lower()}#{spec['instance_id'].split('-')[-1]}")
                            except Exception as save_err:
                                self.logger.error(f"Failed to save image for repository {repo.lower()}#{spec['instance_id'].split('-')[-1]}: {str(save_err)}")

                        except Exception as setup_err:
                            self.logger.error(f"Error configuring environment for repository {repo.lower()}#{spec['instance_id'].split('-')[-1]}: {str(setup_err)}")
                            continue
                    else:
                        container = self.docker_manager.setup_container_and_environment(repo, spec["instance_id"].split("-")[-1])
                    
                    try:
                        self._process_spec(container, spec, repo_name)
                        
                        # Save results immediately
                        spec["processed"] = True
                        self._save_specs(specs_by_repo)
                        self.logger.info(f"Saved results for {spec['instance_id']}")

                    except Exception as inst_err:
                        self.logger.error(f"Error processing {spec['instance_id']}: {str(inst_err)}")
                        
                except Exception as repo_err:
                    self.logger.error(f"Error processing repository {repo}: {str(repo_err)}")
                finally:
                    if container is not None and not self.cleanup_in_progress:
                        self.docker_manager.cleanup_container(container, force_remove=True)

        self.logger.info("All processing completed")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Docker Agent Runner")
    parser.add_argument("--test-only", action="store_true", help="Only run tests, skip environment configuration and image saving")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--agents", nargs="+", help="List of agent names to evaluate")
    parser.add_argument("--max-instances", type=int, default=10, help="Maximum instances per repo")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Import evaluator and run
        from evaluate import AgentEvaluator
        evaluator = AgentEvaluator()
        evaluator.evaluate(agent_names=args.agents, max_instances=args.max_instances)
    else:
        runner = DockerAgentRunner(test_only=args.test_only)
        runner.run()

if __name__ == "__main__":
    main()