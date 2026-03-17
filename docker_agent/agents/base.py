"""Agent base abstract class"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

from docker_agent.core.types import Container
from docker_agent.parsing.patch_analyzer import PatchAnalyzer
from docker_agent.utils.command_executor import DockerCommandExecutor
from docker_agent.core.exceptions import AgentSetupError


class BaseAgent:
    """Agent base abstract class"""

    def __init__(self, container: Container, agent_config):
        self.container = container
        self.agent_config = agent_config
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(__file__).parent.parent
        self.path_analyzer = PatchAnalyzer()
        self.docker_executor = DockerCommandExecutor(container)

    def setup(self):
        """General logic for setting up agent environment"""
        self.logger.info(f"Setting up {self.agent_config.name} environment")

        self._prepare_agent_code()
        self._checkout_branch()
        self._install_dependencies()

        self.logger.info(f"{self.agent_config.name} environment setup completed")

    def _checkout_branch(self):
        """Switch to specified branch"""
        if hasattr(self.agent_config, 'branch') and self.agent_config.branch != "main":
            branch_cmd = f"git checkout {self.agent_config.branch}"
            exit_code, output = self.docker_executor.execute(branch_cmd, "/workdir/agent", stream=True)
            if exit_code != 0:
                self.logger.warning(f"Branch switch failed, continuing with default branch: {output}")

    def _install_dependencies(self):
        """General logic for installing dependencies"""
        if hasattr(self.agent_config, 'install_command') and self.agent_config.install_command:
            self.logger.info(f"Installing {self.agent_config.name} dependencies")
            exit_code, output = self.docker_executor.execute(
                self.agent_config.install_command, "/workdir/agent", stream=True, tty=True, timeout=600
            )
            if exit_code != 0:
                raise AgentSetupError(f"Failed to install agent dependencies: {output}", agent_name=self.agent_config.name)

    @abstractmethod
    def _prepare_agent_code(self):
        """Prepare agent code - must be implemented by subclass (clone or copy)"""
        pass

    @abstractmethod
    def run(self, problem_statement: str, instance_id: str, repo_name: str) -> tuple[bool, str]:
        """Run agent to solve problem - must be implemented by subclass"""
        pass

    @abstractmethod
    def parse_agent_log(self, log: str) -> Dict[str, Optional[int]]:
        """Parse agent log to extract token usage - must be implemented by subclass"""
        pass

    @abstractmethod
    def prepare_resources(self) -> Optional[List[Dict[str, Any]]]:
        """
        Prepare agent-specific resources before evaluation - must be implemented by subclass (clone or copy)

        Returns:
            Agent-specific resources (e.g., agentless patches) or None
        """
        return None