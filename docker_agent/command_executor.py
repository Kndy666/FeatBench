import subprocess
import logging
from typing import Tuple, Optional
import docker.models.containers
from abc import ABC, abstractmethod
import pty
import os
import select
import fcntl
import shutil
import signal

# Get terminal size and define environment variables
terminal_size = shutil.get_terminal_size()
terminal_width = terminal_size.columns
terminal_height = terminal_size.lines

# Define environment variables to inject
PROXY_URL = "http://127.0.0.1:58591"

docker_environment = {
    # "http_proxy": PROXY_URL,
    # "https_proxy": PROXY_URL,
    "COLUMNS": str(terminal_width),
    "LINES": str(terminal_height),
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_OFFLINE": "1"
}

class BaseCommandExecutor(ABC):
    """Command executor base class"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        env = dict(os.environ)
        self.env = env.update(docker_environment)

    def _set_timeout(self, timeout, process=None):
        """Set timeout handling"""
        if timeout is not None:
            def timeout_handler(signum, frame):
                if process is not None:
                    process.terminate()
                raise TimeoutError(f"Timeout {timeout}s")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

    def _cancel_timeout(self, timeout):
        """Cancel timeout handling"""
        if timeout is not None:
            signal.alarm(0)

    @abstractmethod
    def execute(self, command: str, workdir: str, stream: bool = False, tty: bool = True, timeout: Optional[float] = None) -> Tuple[int, str]:
        """Execute command"""
        pass
    
    @abstractmethod
    def _execute_pty(self, command: str, workdir: str, stream: bool, timeout: Optional[float]) -> Tuple[int, str]:
        """Execute command in PTY mode for streaming"""
        pass

    @abstractmethod
    def _execute_without_pty(self, command: str, workdir: str, stream: bool, timeout: Optional[float]) -> Tuple[int, str]:
        """Execute command without PTY mode"""
        pass


class LocalCommandExecutor(BaseCommandExecutor):
    """Local command executor"""
    def __init__(self):
        super().__init__()

    def execute(self, command: str, workdir: str = "/", stream: bool = False, tty: bool = True, timeout: Optional[float] = None) -> Tuple[int, str]:
        """Execute command locally"""
        try:
            if tty:
                return self._execute_pty(command, workdir, stream, timeout)
            else:
                return self._execute_without_pty(command, workdir, stream, timeout)
        except Exception as e:
            self.logger.error(f"Local command execution error: {e}")
            return 1, str(e)

    def _setup_pty_process(self, command: str, workdir: str):
        """Setup PTY process and return (master_fd, slave_fd, process)"""
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=workdir,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=slave_fd,
            preexec_fn=os.setsid,
            env=self.env
        )
        os.close(slave_fd)
        return master_fd, process

    def _execute_pty(self, command: str, workdir: str, stream: bool, timeout: Optional[float]) -> Tuple[int, str]:
        """Execute local command in streaming mode"""
        self.logger.info(f"Local pty execute command: {command}")
        master_fd, process = self._setup_pty_process(command, workdir)
        
        self._set_timeout(timeout, process)
        try:
            if stream:
                # Set non-blocking read
                fcntl.fcntl(master_fd, fcntl.F_SETFL, os.O_NONBLOCK)
                output_lines = []
                
                while True:
                    if process.poll() is not None:
                        try:
                            remaining = os.read(master_fd, 4096).decode('utf-8', errors='replace')
                            if remaining:
                                print(remaining, end='', flush=True)
                                output_lines.append(remaining)
                        except (OSError, BlockingIOError):
                            pass
                        break
                    
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    if ready:
                        try:
                            data = os.read(master_fd, 4096)
                            if data:
                                text = data.decode('utf-8', errors='replace')
                                print(text, end='', flush=True)
                                output_lines.append(text)
                        except (OSError, BlockingIOError):
                            continue
                
                process.wait()
                return process.returncode, ''.join(output_lines)
            else:
                output = b""
                while True:
                    try:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        output += data
                    except OSError:
                        break
                    if process.poll() is not None:
                        try:
                            while True:
                                data = os.read(master_fd, 4096)
                                if not data:
                                    break
                                output += data
                        except OSError:
                            pass
                        break
                process.wait()
                output = output.decode('utf-8', errors='replace')
                print(output, end='', flush=True)
                return process.returncode, output
        finally:
            try:
                os.close(master_fd)
                self._cancel_timeout(timeout)
            except OSError:
                pass
    
    def _execute_without_pty(self, command: str, workdir: str, stream: bool, timeout: Optional[float]) -> Tuple[int, str]:
        """Streaming execution method without PTY"""
        self.logger.info(f"Local execute command: {command}")
        if stream:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=0,  # No buffering
                universal_newlines=True,
                env=self.env
            )
            
            self._set_timeout(timeout, process)
            output_lines = []
            try:
                for line in process.stdout:
                    self.logger.debug(f"Command output: {line.rstrip()}")
                    print(line, end='', flush=True)
                    output_lines.append(line)
                
                process.wait()
                return process.returncode, ''.join(output_lines)
            finally:
                self._cancel_timeout(timeout)
        else:
            if timeout is not None:
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        cwd=workdir,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        timeout=timeout,
                        env=self.env
                    )
                except subprocess.TimeoutExpired:
                    raise TimeoutError(f"Timeout {timeout}s")
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    env=self.env
                )
            output = result.stdout + result.stderr
            print(output, end='', flush=True)
            return result.returncode, output


class DockerCommandExecutor(BaseCommandExecutor):
    """Docker container command executor"""
    
    def __init__(self, container: docker.models.containers.Container):
        super().__init__()
        self.container = container
        self.client = docker.from_env()

    def execute(self, command: str, workdir: str = "/workdir", stream: bool = False, tty: bool = True, timeout: Optional[float] = None) -> Tuple[int, str]:
        """Execute command in Docker container"""
        try:
            if tty:
                return self._execute_pty(command, workdir, stream, timeout)
            else:
                return self._execute_without_pty(command, workdir, stream, timeout)
        except Exception as e:
            self.logger.error(f"Docker command execution error: {e}")
            return 1, str(e)

    def _exec(self, command: str, workdir: str, stream: bool, tty: bool, timeout: Optional[float]) -> Tuple[int, str]:
        """Common execution logic"""
        if timeout is not None:
            timeout_command = f"timeout -s TERM -k 10s {int(timeout)}s {command}"
        else:
            timeout_command = command
            
        exec_instance = self.client.api.exec_create(
            self.container.id,
            cmd=["/bin/bash", "-c", timeout_command],
            workdir=workdir,
            stdout=True,
            stderr=True,
            tty=tty,
            environment=self.env
        )
        output_stream = self.client.api.exec_start(exec_instance['Id'], stream=stream, tty=tty)

        if stream:
            output_lines = []
            for line in output_stream:
                line_str = line.decode('utf-8', errors='replace')
                self.logger.debug(f"Command output: {line_str.rstrip()}")
                print(line_str, end='', flush=True)
                output_lines.append(line_str)

            exit_code = self.client.api.exec_inspect(exec_instance['Id'])['ExitCode']
            if timeout is not None and (exit_code == 124 or exit_code == 137):
                raise TimeoutError(f"Timeout {timeout}s")

            return exit_code, ''.join(output_lines)
        else:
            output = output_stream.decode('utf-8', errors='replace')
            print(output, end='', flush=True)

            exit_code = self.client.api.exec_inspect(exec_instance['Id'])['ExitCode']
            if timeout is not None and (exit_code == 124 or exit_code == 137):
                raise TimeoutError(f"Timeout {timeout}s")

            return exit_code, output

    def _execute_pty(self, command: str, workdir: str, stream: bool, timeout: Optional[float]) -> Tuple[int, str]:
        self.logger.info(f"Docker container pty execute command: {command}")
        return self._exec(command, workdir, stream, True, timeout)

    def _execute_without_pty(self, command: str, workdir: str, stream: bool, timeout: Optional[float]) -> Tuple[int, str]:
        self.logger.info(f"Docker container execute command: {command}")
        return self._exec(command, workdir, stream, False, timeout)