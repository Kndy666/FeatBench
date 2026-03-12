"""Oracle Agent - applies the ground-truth patch from the dataset"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from docker_agent.agents.base import BaseAgent
from docker_agent.parsing.patch_analyzer import PatchInfo


class OracleAgent(BaseAgent):
    """
    Oracle agent that applies the ground-truth patch directly from the dataset.

    This agent does not run any LLM; instead it applies ``spec.patch``
    (a list of GitHub-style file-change dicts) to the repository and then
    captures the result with ``git diff``, exactly like the other agents
    produce their ``patch.diff`` artefact.  It serves as an upper-bound
    reference when evaluating the benchmark.

    Usage in manager.py:
        ``AgentManager.evaluate()`` sets ``agent.spec_patch = spec.patch``
        before calling ``agent.run()``.
    """

    def __init__(self, container, agent_config):
        super().__init__(container, agent_config)
        # Set by AgentManager.evaluate() before run() is called.
        self.spec_patch: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _prepare_agent_code(self):
        """No external agent code to install."""
        self.logger.info("OracleAgent: no agent code to prepare")

    def run(self, problem_statement: str, instance_id: str, repo_name: str) -> tuple[bool, str]:
        """Apply the ground-truth patch and capture a git diff as patch.diff."""
        self.logger.info(f"OracleAgent: applying ground-truth patch for {instance_id}")

        if not self.spec_patch:
            msg = "OracleAgent.run() called without spec_patch being set"
            self.logger.error(msg)
            return False, msg

        repo_workdir = f"/workdir/swap/{repo_name}"
        patch_path = f"{repo_workdir}/patch.diff"

        try:
            # Build PatchInfo objects from the GitHub-style diff dicts.
            patches: List[PatchInfo] = []
            for change in self.spec_patch:
                filename = change.get("filename")
                patch_content = change.get("patch", "")
                status = change.get("status", "")

                if not filename or not patch_content or not status:
                    continue

                patches.append(PatchInfo(
                    filename=filename,
                    status=status,
                    patch_content=patch_content,
                    is_test_file=self.path_analyzer.is_test_file(filename),
                ))

            if not patches:
                msg = "OracleAgent: no valid patches found in spec_patch"
                self.logger.warning(msg)
                return False, msg

            applied = self.path_analyzer.apply_patches_to_container(
                patches, self.docker_executor, repo_workdir
            )
            self.logger.info(f"OracleAgent: applied patches to {len(applied)} file(s): {applied}")

            # Produce the patch.diff artefact that the evaluator expects.
            diff_exit, diff_output = self.docker_executor.execute(
                f"git diff > {patch_path}", repo_workdir, stream=True
            )
            if diff_exit != 0:
                self.logger.warning(f"OracleAgent: git diff failed: {diff_output}")
                return False, diff_output

            return True, f"Applied {len(applied)} patch(es): {applied}"

        except Exception as e:
            self.logger.error(f"OracleAgent error: {e}")
            return False, str(e)

    def parse_agent_log(self, log: str) -> Dict[str, Optional[int]]:
        """Oracle generates no LLM tokens."""
        return {"Total Tokens": None, "Input Tokens": None, "Output Tokens": None}

    def prepare_resources(self) -> Optional[List[Dict[str, Any]]]:
        """No pre-computed resources needed."""
        return None
