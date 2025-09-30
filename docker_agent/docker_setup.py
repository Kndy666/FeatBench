import docker
import os
import logging
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
import docker.models.containers
from command_executor import LocalCommandExecutor, DockerCommandExecutor, docker_environment
from docker_image_builder import DockerImageBuilder
from locate_test import CodeChange
from pytest_output_parse import TestStatus, PytestResultParser
from patch_analyzer import PatchAnalyzer, PatchInfo
from abc import ABC, abstractmethod

class CacheManager:
    """Container and image cache manager"""
    
    def __init__(self, repo: str, repo_id: str, timeout=300):
        self.logger = logging.getLogger(__name__)
        self.client = docker.from_env(timeout=timeout)
        self.repo = repo.replace("/", "_")
        self.repo_id = repo_id
        self.repo_lower = self.repo.lower()
        self.image_builder = DockerImageBuilder(timeout)
        self.base_path = Path(__file__).parent

    @property
    def common_container_config(self) -> Dict[str, Any]:
        """Extract and return common container creation parameters"""
        
        config = {
            "name": self.repo,
            "command": "/bin/bash",
            "detach": True,
            "tty": True,
            "runtime": "nvidia",
            "network_mode": "host",
            "device_requests": [{
                'count': -1,
                'capabilities': [['gpu']]
            }],
            "environment": docker_environment,
            "volumes": {
                str(self.base_path / "swap"): {
                    "bind": "/workdir/swap",
                    "mode": "rw"
                }
            }
        }

        if os.name == 'posix':
            uid = os.getuid()
            gid = os.getgid()
            self.logger.info(f"Running on POSIX system, setting container user to UID={uid}, GID={gid}")
            config['user'] = f"{uid}:{gid}"
            
        return config
    
    def check_cached_container(self) -> Optional[docker.models.containers.Container]:
        """Check if cached container exists"""
        
        try:
            # Find existing containers
            container = self.client.containers.get(self.repo)
            
            # Check container status
            if container.status == 'running':
                self.logger.info(f"Found running cached container: {self.repo}")
                return container
            elif container.status == 'exited':
                self.logger.info(f"Found stopped cached container: {self.repo}, restarting...")
                container.start()
                return container
            else:
                self.logger.warning(f"Container {self.repo} status abnormal: {container.status}, will recreate")
                container.remove(force=True)
                return None
                
        except docker.errors.NotFound:
            self.logger.info(f"Cached container not found: {self.repo}")
            return None
        except Exception as e:
            self.logger.error(f"Error checking cached container: {str(e)}")
            return None

    def save_container_as_image(self, container: docker.models.containers.Container) -> str:
        """Save container as new image"""

        # Image name must be lowercase
        image_name = f"cached_{self.repo_lower}"
        
        try:
            self.logger.info(f"Saving container as image: {image_name}")
            
            # Commit container as new image
            image = container.commit(repository=image_name, tag=self.repo_id)
            
            self.logger.info(f"Successfully saved image: {image_name}:latest (ID: {image.id[:12]})")
            return image.id
            
        except Exception as e:
            self.logger.error(f"Failed to save container image: {str(e)}")
            raise RuntimeError(f"Failed to save container image: {str(e)}")

    def check_cached_image(self) -> bool:
        """Check if cached image exists"""

        image_name = f"cached_{self.repo_lower}:{self.repo_id}"
        
        try:
            self.client.images.get(image_name)
            self.logger.info(f"Found cached image: {image_name}")
            return True
        except docker.errors.ImageNotFound:
            self.logger.info(f"Cached image not found: {image_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error checking cached image: {str(e)}")
            return False

    def create_container_from_cached_image(self) -> docker.models.containers.Container:
        """Create container from cached image"""

        image_name = f"cached_{self.repo_lower}:{self.repo_id}"
        
        self.logger.info(f"Creating container from cached image: {image_name}")
        
        container = self.client.containers.run(
            image=image_name,
            **self.common_container_config
        )
        
        self.logger.info(f"Successfully created container from cached image: {self.repo}")
        return container

    def create_new_container(self) -> docker.models.containers.Container:
        """Create new container"""
        self.logger.info(f"Creating new container: {self.repo}")

        # Build dynamic image
        image_name = self.image_builder.build_image(self.repo)

        # Create container with GPU support
        container = self.client.containers.run(
            image=image_name,
            **self.common_container_config
        )

        self.logger.info(f"Container {self.repo} created successfully (with GPU support)")
        return container

class ContainerOperator:
    """Container operator class"""
    
    def __init__(self, repo: str, container: Optional[docker.models.containers.Container] = None):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.docker_executor = DockerCommandExecutor(container)
        self.local_executor = LocalCommandExecutor()
        self.base_path = Path(__file__).parent
        self.repo = repo
        self.repo_name = repo.split("/")[-1]
        self.patch_analyzer = PatchAnalyzer()

        if self.container:
            self.docker_executor.execute(f"git config --global --add safe.directory /workdir/swap/{self.repo_name}")

    def repo_clone(self, use_docker=True):
        """Clone repository"""
        # Check if directory already exists
        if use_docker:
            check_cmd = f"test -d swap/{self.repo_name}"
            exit_code, _ = self.docker_executor.execute(check_cmd)
        else:
            repo_path = self.base_path / "swap" / self.repo_name
            if repo_path.exists():
                exit_code = 0
            else:
                exit_code = 1
        
        if exit_code == 0:
            self.logger.info(f"Directory {self.repo_name} already exists, skipping clone")
            return
        
        repo_url = f"https://github.com/{self.repo}.git"
        command = f"git clone {repo_url}"
        
        # Use streaming command execution
        if use_docker:
            exit_code, output = self.docker_executor.execute(command, "/workdir/swap", stream=True, tty=True)
        else:
            exit_code, output = self.local_executor.execute(command, self.base_path / "swap", stream=True, tty=True)

        self.logger.info(f"Command completed, return code: {exit_code}")
        if exit_code is not None and exit_code != 0:
            self.logger.error(f"Command execution failed: {command}\nError: {output}")
            raise RuntimeError(f"Command execution failed: {command}\nError: {output}")

    def checkout_commit(self, commit_hash: str, exclude_file: List[str] = None, use_docker=True) -> None:
        """Switch to specified commit"""
        self.logger.info(f"Forcibly switching to commit: {commit_hash}")
        if exclude_file is None:
            exclude_file = []
        commands = [
            "git reset --hard",
            "git clean -fd " + " ".join([f"-e {f}" for f in exclude_file]),
            f"git checkout {commit_hash}"
        ]
        
        for cmd in commands:
            if use_docker:
                exit_code, output = self.docker_executor.execute(cmd, str(Path("/workdir/swap") / self.repo_name), tty=False, timeout=30)
            else:
                exit_code, output = self.local_executor.execute(cmd, self.base_path / "swap" / self.repo_name, tty=False, timeout=30)

            if exit_code != 0:
                self.logger.error(f"Command execution failed: {cmd}\nError: {output}")
                raise RuntimeError(f"Command execution failed: {cmd}\nError: {output}")
                
            self.logger.info(f"Execution successful: {cmd.split('&&')[-1].strip()}")
        
        self.logger.info(f"Successfully forcibly switched to commit: {commit_hash}")

    def apply_patches(self, file_changes: List[Dict]) -> List[str]:
        """Apply file changes - compatible with original interface, using unified patch analyzer"""
        # Convert original format to PatchInfo format
        patches = []
        for change in file_changes:
            filename = change.get("filename")
            patch_content = change.get("patch", "")
            status = change.get("status", "")
            
            if not filename or not patch_content or not status:
                continue
            
            patch_info = PatchInfo(
                filename=filename,
                status=status,
                patch_content=patch_content,
                is_test_file=self.patch_analyzer.is_test_file(filename)
            )
            patches.append(patch_info)

        workdir = str(Path("/workdir/swap") / self.repo_name)
        return self.patch_analyzer.apply_patches_to_container(patches, self.docker_executor, workdir)

    def _find_test_dirs(self, repo_name: str, use_docker: bool = True) -> List[str]:
        """Recursively detect test directories in repository (in container or locally), return list of existing directories (if not detected return ['tests'])"""
        candidates = ["tests", "test", "Tests", "TESTS", "unit_tests", "TEST"]
        ignore_dirs = [".venv", "build"]

        # First search in root directory
        root_find_cmd = (
            "find . -maxdepth 1 -type d \\( " +
            " -o ".join([f"-name '{d}'" for d in candidates]) +
            " \\) -print"
        )

        if use_docker:
            workdir = f"/workdir/swap/{repo_name}"
            exit_code, output = self.docker_executor.execute(root_find_cmd, workdir, tty=False, timeout=30)
        else:
            workdir = str(self.base_path / "swap" / repo_name)
            exit_code, output = self.local_executor.execute(root_find_cmd, workdir, tty=False, timeout=30)

        if output is None:
            output = ""

        # Clean paths, remove leading ./
        found = [line.strip().lstrip('./') for line in output.splitlines() if line.strip()]

        # If test directories found in root directory, return directly
        if found:
            self.logger.info(f"Test directories detected in root directory: {found}")
            return found

        # Root directory not found, continue recursive search
        prune_expr = " -o ".join([f"-path './{d}' -prune" for d in ignore_dirs])
        prune_expr = f"\\( {prune_expr} \\) -o "

        find_cmd = (
            f"find . {prune_expr}-type d \\( " +
            " -o ".join([f"-name '{d}'" for d in candidates]) +
            " \\) -print"
        )

        if use_docker:
            exit_code, output = self.docker_executor.execute(find_cmd, workdir, tty=False, timeout=30)
        else:
            exit_code, output = self.local_executor.execute(find_cmd, workdir, tty=False, timeout=30)

        if output is None:
            output = ""

        found = [line.strip().lstrip('./') for line in output.splitlines() if line.strip()]

        if not found:
            self.logger.info(f"Common test directories not detected ({candidates}), falling back to default 'tests'")
            return ["tests"]

        self.logger.info(f"Test directories detected recursively: {found}")
        return found

    def run_tests_in_container(
        self,
        repo_name: str,
        test_files: Optional[List[Dict[str, CodeChange] | str]] = None,
        expected_statuses: Optional[List[TestStatus]] = None,
        use_xdist: bool = True
    ) -> tuple[Set[str], str]:
        """Run tests in container and return passed test files and logs"""
        pytest_args = []

        if test_files is None:
            dirs = self._find_test_dirs(repo_name, use_docker=True)
            for d in dirs:
                pytest_args.append(f"{d}/")
        else:
            if isinstance(test_files[0], Dict):
                for test_file in test_files:
                    for file_name, changes in test_file.items():
                        for change in changes:
                            if change.change_type == 'deleted':
                                continue
                            elif change.code_type == 'function':
                                pytest_args.append(f"{file_name}::{change.name}")
                            elif change.code_type == 'method':
                                class_name, method_name = change.name.split('.', 1)
                                pytest_args.append(f"{file_name}::{class_name}::{method_name}")
            else:
                pytest_args.extend(test_files)
        
        # Check command length, if too long use batch execution directly
        base_cmd_template = "python3 -m pytest -q -rA --tb=no -p no:pretty --timeout=5 --continue-on-collection-errors"
        if use_xdist:
            base_cmd_template = f"pip install pytest-xdist && {base_cmd_template} --timeout-method=thread -n auto"
        else:
            base_cmd_template = f"{base_cmd_template} --timeout-method=signal"
        
        # Estimate full command length (conservative estimate bash limit 100KB)
        estimated_length = len(base_cmd_template) + sum(len(arg) + 1 for arg in pytest_args)

        if estimated_length > 100000:  # If exceeds 100KB, use batch execution directly
            self.logger.info(f"Too many test parameters ({len(pytest_args)}), using batch execution")
            return self._run_tests_in_batches(repo_name, pytest_args, base_cmd_template, expected_statuses)
        
        # Command length reasonable, execute directly
        cmd = f"{base_cmd_template} {' '.join(pytest_args)}"

        exit_code, output = self.docker_executor.execute(
            cmd, f"/workdir/swap/{repo_name}", stream=True, tty=True, timeout=1200
        )
        matched_files = self.parse_pytest_output(output, pytest_args, expected_statuses)
        return matched_files, output

    def _run_tests_in_batches(self, repo_name: str, pytest_args: List[str], base_cmd_template: str, expected_statuses: Optional[List[TestStatus]] = None) -> tuple[Set[str], str]:
        """When command is too long, execute tests in batches"""
        self.logger.info("Executing tests in batches to avoid command length limit")

        batch_size = 250  # Max 250 tests per batch
        all_output = []
        all_matched = set()
        
        for i in range(0, len(pytest_args), batch_size):
            batch = pytest_args[i:i + batch_size]
            self.logger.info(f"Executing batch {i//batch_size + 1} of tests ({len(batch)})")
            
            cmd = f"{base_cmd_template} {' '.join(batch)}"
            exit_code, output = self.docker_executor.execute(
                cmd, f"/workdir/swap/{repo_name}", stream=True, tty=True, timeout=1200
            )
            
            all_output.append(output)
            batch_matched = self.parse_pytest_output(output, batch, expected_statuses)
            all_matched.update(batch_matched)
        
        combined_output = '\n'.join(all_output)
        return all_matched, combined_output

    def parse_pytest_output(self, logs: str, test_cases: List[str], expected_statuses: List[TestStatus]) -> Set[str]:
        """Parse pytest output, extract files with completely passed tests (no failures or errors)"""
        
        parser = PytestResultParser(logs)
        
        # Check if test_cases are directory format
        is_directory_test = any(arg.endswith('/') for arg in test_cases)
        
        if is_directory_test:
            # Use parser's filter function to get all test items matching expected status
            matched = parser.filter_tests_by_status(expected_statuses)
            self.logger.info(f"Directory test matched {len(matched)} tests with expected status")
            return matched
        else:
            # Original processing logic
            results = parser.query_tests(test_cases)
            self.logger.info("Query results:")
            for test, status in results.items():
                self.logger.info(f"  {test}: {status.value}")
            return set(test for test, status in results.items() if status in expected_statuses)

class DockerEnvironmentManager:
    """Docker environment manager"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def setup_container_and_environment(self, repo: str, repo_id: str, timeout=300) -> docker.models.containers.Container:
        """Create Docker container and configure test environment (with cache support)"""

        self.cache_manager = CacheManager(repo, repo_id, timeout)
        # First check if cached container exists
        cached_container = self.cache_manager.check_cached_container()
        if cached_container:
            return cached_container

        # Check if cached image exists
        if self.cache_manager.check_cached_image():
            return self.cache_manager.create_container_from_cached_image()

        # Create new container (using dynamically built image)
        return self.cache_manager.create_new_container()
    
    def cleanup_container(self, container: docker.models.containers.Container, force_remove: bool = False) -> None:
        """Clean up container resources"""
        if container:
            try:
                if force_remove:
                    container.stop()
                    container.remove()
                    self.logger.info(f"Container {container.name} has been deleted")
                else:
                    self.logger.info(f"Container {container.name} retained as cache")
                    
            except Exception as e:
                self.logger.error(f"Error handling container: {str(e)}")

class BaseAgent(ABC):
    """Agent base abstract class"""
    
    def __init__(self, container: docker.models.containers.Container, agent_config):
        self.container = container
        self.agent_config = agent_config
        self.logger = logging.getLogger(__name__)
        self.docker_executor = DockerCommandExecutor(container)
    
    def setup_agent(self):
        """General logic for setting up agent environment"""
        self.logger.info(f"Setting up {self.agent_config.name} environment")
        
        # Prepare agent code
        self._prepare_agent_code()
        
        # Switch branch (if needed)
        self._checkout_branch()
        
        # Set environment variables
        self._setup_environment_variables()
        
        # Install dependencies
        self._install_dependencies()
        
        self.logger.info(f"{self.agent_config.name} environment setup completed")
    
    @abstractmethod
    def _prepare_agent_code(self):
        """Prepare agent code - must be implemented by subclass (clone or copy)"""
        pass
    
    def _checkout_branch(self):
        """Switch to specified branch"""
        if hasattr(self.agent_config, 'branch') and self.agent_config.branch != "main":
            branch_cmd = f"git checkout {self.agent_config.branch}"
            exit_code, output = self.docker_executor.execute(branch_cmd, "/workdir/agent", stream=True)
            if exit_code != 0:
                self.logger.warning(f"Branch switch failed, continuing with default branch: {output}")
    
    def _setup_environment_variables(self):
        """Set environment variables"""
        if hasattr(self.agent_config, 'extra_env'):
            for key, value in self.agent_config.extra_env.items():
                # Support environment variable replacement
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    actual_value = os.environ.get(env_var, "")
                    if actual_value:
                        export_cmd = f"export {key}='{actual_value}'"
                        self.docker_executor.execute(export_cmd, "/workdir/agent", tty=False)
                else:
                    export_cmd = f"export {key}='{value}'"
                    self.docker_executor.execute(export_cmd, "/workdir/agent", tty=False)

    def _install_dependencies(self):
        """General logic for installing dependencies"""
        if hasattr(self.agent_config, 'install_command') and self.agent_config.install_command:
            self.logger.info(f"Installing {self.agent_config.name} dependencies")
            exit_code, output = self.docker_executor.execute(
                self.agent_config.install_command, "/workdir/agent", stream=True, tty=True, timeout=600
            )
            if exit_code != 0:
                raise RuntimeError(f"Failed to install agent dependencies: {output}")
    
    @abstractmethod
    def run_agent_on_problem(self, problem_statement: str, instance_id: str, repo_name: str) -> tuple[bool, str]:
        """Run agent on problem - must be implemented by subclass"""
        pass

class TraeAgent(BaseAgent):
    """Specific implementation of Trae-Agent"""

    def _prepare_agent_code(self):
        """Clone trae-agent repository"""
        clone_cmd = f"git clone {self.agent_config.repo_url} agent/"
        exit_code, output = self.docker_executor.execute(clone_cmd, "/workdir", stream=True, timeout=300)
            
        if exit_code != 0:
            raise RuntimeError(f"Failed to clone agent repository: {output}")
    
    def run_agent_on_problem(self, problem_statement: str, instance_id: str, repo_name: str) -> tuple[bool, str]:
        """Run trae-agent to solve problem"""
        self.logger.info(f"Running {self.agent_config.name} to solve problem {instance_id}")
        
        try:
            escaped_problem = shlex.quote(problem_statement)
            run_cmd = self._build_command(escaped_problem, repo_name)
            
            # Run agent
            exit_code, agent_output = self.docker_executor.execute(
                run_cmd, "/workdir/agent", stream=True, tty=True
            )

            success = exit_code == 0
            return success, agent_output
            
        except Exception as e:
            self.logger.error(f"Error running trae-agent: {str(e)}")
            return False, str(e)
    
    def _build_command(self, escaped_problem: str, repo_name: str) -> str:
        """Build trae-agent run command"""
        return (".venv/bin/python3.12 -m trae_agent.cli run "
            f"{escaped_problem} "
            "--must-patch "
            f"--patch-path /workdir/swap/{repo_name}/patch.diff "
            f"--working-dir /workdir/swap/{repo_name} "
            f"--model {self.agent_config.model} "
            f"--provider {self.agent_config.provider} "
            f"--config-file /workdir/swap/trae-agent/trae_config.yaml")

class Agentless(BaseAgent):
    """Specific implementation of Agentless"""
    
    def _prepare_agent_code(self):
        """Copy local agentless code to container"""
        local_path = self.agent_config.repo_url  # Relative path, like "../agentless"
        
        # Create agent directory
        self.docker_executor.execute("mkdir -p agent/", "/workdir", stream=True)
        
        # Copy local agentless code to container
        try:
            import tarfile
            import io
            
            # Create tar package
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                # Add agentless directory to tar package
                base_path = Path(__file__).parent
                agentless_path = base_path / local_path
                
                if not agentless_path.exists():
                    raise FileNotFoundError(f"Local agentless path does not exist: {agentless_path}")
                
                # Recursively add all files
                for file_path in agentless_path.rglob('*'):
                    if file_path.is_file():
                        # Calculate relative path
                        relative_path = file_path.relative_to(agentless_path)
                        tar.add(file_path, arcname=str(relative_path))
            
            tar_stream.seek(0)
            
            # Copy to container
            success = self.container.put_archive("/workdir/agent", tar_stream.getvalue())
            if not success:
                raise RuntimeError("Failed to copy agentless code to container")
                
            self.logger.info(f"Successfully copied local agentless code to container: {local_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to copy agentless code: {str(e)}")
            raise RuntimeError(f"Failed to copy agentless code: {str(e)}")
    
    def run_agent_on_problem(self, model_patch: str, instance_id: str, repo_name: str) -> tuple[bool, str]:
        """Run agentless to solve problem - to be implemented"""
        raise NotImplementedError("Agentless run_agent_on_problem method not yet implemented")

class AgentManager:
    """Agent manager, responsible for setting up and running different agents in container"""
    
    def __init__(self, container: docker.models.containers.Container, agent_config):
        self.container = container
        self.agent_config = agent_config
        self.logger = logging.getLogger(__name__)
        self.agent = self._create_agent()
    
    def _create_agent(self) -> BaseAgent:
        """Create corresponding agent instance based on configuration"""
        agent_name = self.agent_config.name.lower()
        
        if agent_name == "trae-agent":
            return TraeAgent(self.container, self.agent_config)
        elif agent_name == "agentless":
            return Agentless(self.container, self.agent_config)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_config.name}")
    
    def setup_agent(self):
        """Set up agent environment"""
        self.agent.setup_agent()
    
    def run_agent_on_problem(self, problem_statement: str, instance_id: str, repo_name: str) -> tuple[bool, str]:
        """Run agent on problem"""
        return self.agent.run_agent_on_problem(problem_statement, instance_id, repo_name)