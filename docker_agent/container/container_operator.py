"""Container operator class"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Set

from docker_agent.core.types import TestStatus, CodeChange, Container
from docker_agent.parsing.patch_analyzer import PatchAnalyzer, PatchInfo
from docker_agent.parsing.pytest_parser import PytestResultParser
from docker_agent.utils.command_executor import LocalCommandExecutor, DockerCommandExecutor
from docker_agent.core.exceptions import ContainerOperationError


class ContainerOperator:
    """Container operator class"""

    def __init__(self, repo: str, container: Optional[Container] = None):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.docker_executor = DockerCommandExecutor(container)
        self.local_executor = LocalCommandExecutor()
        self.base_path = Path(__file__).parent.parent  # Go up to the root
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

        if use_docker:
            exit_code, output = self.docker_executor.execute(command, "/workdir/swap", stream=True, tty=True)
        else:
            exit_code, output = self.local_executor.execute(command, self.base_path / "swap", stream=True, tty=True)

        self.logger.info(f"Command completed, return code: {exit_code}")
        if exit_code is not None and exit_code != 0:
            self.logger.error(f"Command execution failed: {command}\nError: {output}")
            raise ContainerOperationError(f"Command execution failed: {command}\nError: {output}", container_id=self.container.id if self.container else None)

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
                raise ContainerOperationError(f"Command execution failed: {cmd}\nError: {output}", container_id=self.container.id if self.container else None)

            self.logger.info(f"Execution successful: {cmd.split('&&')[-1].strip()}")

        self.logger.info(f"Successfully forcibly switched to commit: {commit_hash}")

    def apply_patches(self, file_changes: List[Dict]) -> List[str]:
        """Apply file changes - compatible with original interface, using unified patch analyzer"""
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

    def _setup_conan_cmake_env(self, repo_name: str) -> None:
        """Prepare the cmake environment expected by conan's test/conftest.py.

        Conan's conftest.py has hardcoded Linux paths per cmake version. This:
        1. Symlinks the system cmake binary to all configured paths so any-version tests pass.
        2. Installs cmake 3.15.7 (the default version in conftest.py) if not already present,
           so version-checking tests pass. Skipped entirely if the sentinel file exists.
        """
        sentinel = "/usr/share/cmake-3.15.7/.conan_setup_done"
        cmake_315_bin = "/usr/share/cmake-3.15.7/bin"
        script = (
            f"if [ -f {sentinel} ]; then exit 0; fi && "
            # Capture the current system cmake (before potentially installing 3.15)
            "cmake_bin=$(which cmake 2>/dev/null) && "
            "[ -n \"$cmake_bin\" ] && "
            # Symlink system cmake to all configured paths except the 3.15 path
            "python3 -c \""
            "import re; "
            "f=open('test/conftest.py'); c=f.read(); f.close(); "
            "paths=re.findall(r\\\"'Linux': ['\\\\\\\"]([^'\\\\\\\"]+)['\\\\\\\"]\\\", c); "
            "print('\\\\n'.join(paths))"
            f"\" | while read path; do "
            f"[ \"$path\" != \"$(dirname $cmake_bin)\" ] && [ \"$path\" != \"{cmake_315_bin}\" ] && mkdir -p \"$path\" && ln -sf \"$cmake_bin\" \"$path/cmake\"; "
            "done && "
            # Install cmake 3.15.7 only if system cmake is not already 3.15.x
            "cmake_ver=$(cmake --version 2>/dev/null | head -1) && "
            "if ! echo \"$cmake_ver\" | grep -q 'cmake version 3\\.15'; then "
            "  wget -q https://cmake.org/files/v3.15/cmake-3.15.7-Linux-x86_64.sh -O /tmp/cmake-3.15.sh && "
            "  chmod +x /tmp/cmake-3.15.sh && "
            "  /tmp/cmake-3.15.sh --prefix=/usr/local --skip-license && "
            f"  mkdir -p {cmake_315_bin} && "
            f"  ln -sf /usr/local/bin/cmake {cmake_315_bin}/cmake; "
            "else "
            f"  mkdir -p {cmake_315_bin} && ln -sf \"$cmake_bin\" \"{cmake_315_bin}/cmake\"; "
            "fi && "
            f"touch {sentinel}"
        )
        workdir = f"/workdir/swap/{repo_name}"
        exit_code, output = self.docker_executor.execute(script, workdir, tty=False, timeout=120)
        if exit_code != 0:
            self.logger.warning(f"conan cmake env setup failed (non-fatal): {output}")
        else:
            self.logger.info("conan cmake env set up successfully")

    def run_tests_in_container(
        self,
        repo_name: str,
        test_files: Optional[List[Dict[str, CodeChange] | str]] = None,
        expected_statuses: Optional[List[TestStatus]] = None,
        use_xdist: bool = True
    ) -> tuple[Set[str], str]:
        """Run tests in container and return passed test files and logs"""
        if repo_name == "conan":
            self._setup_conan_cmake_env(repo_name)

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

        base_cmd_template = "python3 -m pytest -q -rA --tb=no -p no:pretty --timeout=5 --continue-on-collection-errors"
        if use_xdist:
            base_cmd_template = f"{base_cmd_template} --timeout-method=thread -n auto"
        else:
            base_cmd_template = f"{base_cmd_template} --timeout-method=signal"

        # Estimate full command length (conservative estimate bash limit 100KB)
        estimated_length = len(base_cmd_template) + sum(len(arg) + 1 for arg in pytest_args)

        if estimated_length > 100000:  # If exceeds 100KB, use batch execution directly
            self.logger.info(f"Too many test parameters ({len(pytest_args)}), using batch execution")
            return self._run_tests_in_batches(repo_name, pytest_args, base_cmd_template, expected_statuses)

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

        is_directory_test = any(arg.endswith('/') for arg in test_cases)

        if is_directory_test:
            matched = parser.filter_tests_by_status(expected_statuses)
            self.logger.info(f"Directory test matched {len(matched)} tests with expected status")
            return matched
        else:
            results = parser.query_tests(test_cases)
            self.logger.info("Query results:")
            for test, status in results.items():
                self.logger.info(f"  {test}: {status.value}")
            return set(test for test, status in results.items() if status in expected_statuses)
