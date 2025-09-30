import re
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class PatchInfo:
    """Patch information for a single file"""
    filename: str
    status: str  # added, modified, removed, renamed
    patch_content: str
    is_test_file: bool = False
    old_filename: Optional[str] = None  # For renamed files

class PatchAnalyzer:
    """Unified patch analyzer and applier"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common test file patterns
        self.test_patterns = [
            r'test.*\.py$',
            r'.*test\.py$', 
            r'.*_test\.py$',
            r'.*/test[s]?/.*\.py$',
            r'.*/testing/.*\.py$',
        ]
    
    def is_test_file(self, filename: str) -> bool:
        """Determine if file is a test file"""
        filename_lower = filename.lower()
        return any(re.search(pattern, filename_lower) for pattern in self.test_patterns)
    
    def parse_unified_diff(self, diff_content: str) -> List[PatchInfo]:
        """Parse unified diff format, return patch information for each file"""
        patches = []
        
        # Split into individual file diffs
        file_diffs = re.split(r'\ndiff --git', diff_content)
        
        for i, file_diff in enumerate(file_diffs):
            if i > 0:  # First split may not start with diff
                file_diff = 'diff --git' + file_diff
            
            if not file_diff.strip():
                continue
                
            patch_info = self._parse_single_file_diff(file_diff)
            if patch_info:
                patches.append(patch_info)
        
        self.logger.info(f"Parsed {len(patches)} file patches, test files: {sum(1 for p in patches if p.is_test_file)}")
        return patches
    
    def _parse_single_file_diff(self, diff_content: str) -> Optional[PatchInfo]:
        """Parse single file diff"""
        lines = diff_content.strip().split('\n')
        
        if not lines:
            return None
        
        # Parse file path and status
        git_line = lines[0]
        filename, status, old_filename = self._extract_file_info(git_line, lines)
        
        if not filename:
            return None
        
        # Extract patch content (excluding header information)
        patch_lines = []
        in_hunk = False
        
        for line in lines:
            if line.startswith('@@'):
                in_hunk = True
                patch_lines.append(line)
            elif in_hunk and (line.startswith(('+', '-', ' ')) or line == ''):
                patch_lines.append(line)
            elif line.startswith('\\'):  # Handle "No newline at end of file"
                patch_lines.append(line)
        
        patch_content = '\n'.join(patch_lines)
        is_test = self.is_test_file(filename)
        
        return PatchInfo(
            filename=filename,
            status=status,
            patch_content=patch_content,
            is_test_file=is_test,
            old_filename=old_filename
        )
    
    def _extract_file_info(self, git_line: str, all_lines: List[str]) -> Tuple[Optional[str], str, Optional[str]]:
        """Extract file information from git diff line and related lines"""
        # Parse diff --git a/file b/file
        git_match = re.match(r'diff --git a/(.*?) b/(.*)', git_line)
        if not git_match:
            return None, "unknown", None
        
        old_file, new_file = git_match.groups()
        
        # Check file status
        status = "modified"
        old_filename = None
        
        for line in all_lines[:10]:  # Only check first few lines
            if line.startswith('new file mode'):
                status = "added"
                break
            elif line.startswith('deleted file mode'):
                status = "removed"
                break
            elif line.startswith('rename from'):
                status = "renamed"
                old_filename = old_file
                break
        
        filename = new_file if status != "removed" else old_file
        return filename, status, old_filename
    
    def read_patch_file(self, patch_path: Union[str, Path]) -> str:
        """Read patch file content"""
        patch_path = Path(patch_path)
        
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file does not exist: {patch_path}")
        
        try:
            with patch_path.open('r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"Successfully read patch file: {patch_path}")
            return content
        except Exception as e:
            raise RuntimeError(f"Failed to read patch file: {e}")
    
    def filter_patches(self, patches: List[PatchInfo], include_test: bool = True, 
                      include_source: bool = True) -> List[PatchInfo]:
        """Filter patch list"""
        filtered = []
        
        for patch in patches:
            if patch.is_test_file and include_test:
                filtered.append(patch)
            elif not patch.is_test_file and include_source:
                filtered.append(patch)
        
        self.logger.info(f"After filtering, {len(filtered)} patches retained (test files: {include_test}, source files: {include_source})")
        return filtered
    
    def apply_patches_to_container(self, patches: List[PatchInfo], docker_executor, workdir: str) -> List[str]:
        """Apply patch list in container"""
        applied_files = []
        
        for patch in patches:
            try:
                success = self._apply_single_patch_to_container(patch, docker_executor, workdir)
                if success:
                    applied_files.append(patch.filename)
                    self.logger.info(f"Successfully applied patch: {patch.filename} ({patch.status})")
                else:
                    self.logger.warning(f"Failed to apply patch: {patch.filename}")
            except Exception as e:
                self.logger.error(f"Error applying patch {patch.filename}: {e}")
        
        return applied_files
    
    def _apply_single_patch_to_container(self, patch: PatchInfo, docker_executor, workdir: str) -> bool:
        """Apply single patch in container"""
        # Build complete diff content
        diff_content = self._build_complete_diff(patch)
        
        # Encode patch content to base64 and write to temporary file
        patch_base64 = base64.b64encode(diff_content.encode('utf-8')).decode('utf-8')
        write_cmd = f"echo '{patch_base64}' | base64 -d > /tmp/single_patch.tmp"
        
        exit_code, output = docker_executor.execute(write_cmd, tty=False, timeout=30)
        if exit_code != 0:
            self.logger.error(f"Failed to write patch to temporary file: {output}")
            return False
        
        # Apply patch (use parameters to avoid interaction)
        apply_cmd = "patch -p1 --no-backup-if-mismatch --force < /tmp/single_patch.tmp"
        exit_code, output = docker_executor.execute(apply_cmd, workdir, tty=False, timeout=30)
        
        if exit_code != 0:
            self.logger.error(f"Failed to apply patch: {output}")
            return False
        
        return True
    
    def _build_complete_diff(self, patch: PatchInfo) -> str:
        """Build complete diff format content"""
        header = f"diff --git a/{patch.filename} b/{patch.filename}\n"
        
        if patch.status == "added":
            diff_content = (
                f"{header}"
                f"new file mode 100644\n"
                f"index 0000000..1111111\n"
                f"--- /dev/null\n"
                f"+++ b/{patch.filename}\n"
                f"{patch.patch_content}\n"
            )
        elif patch.status == "removed":
            diff_content = (
                f"{header}"
                f"deleted file mode 100644\n"
                f"index 1111111..0000000\n"
                f"--- a/{patch.filename}\n"
                f"+++ /dev/null\n"
                f"{patch.patch_content}\n"
            )
        elif patch.status == "renamed":
            old_name = patch.old_filename or patch.filename
            diff_content = (
                f"diff --git a/{old_name} b/{patch.filename}\n"
                f"similarity index 100%\n"
                f"rename from {old_name}\n"
                f"rename to {patch.filename}\n"
                f"{patch.patch_content}\n"
            )
        else:  # modified
            diff_content = (
                f"{header}"
                f"index 1111111..2222222 100644\n"
                f"--- a/{patch.filename}\n"
                f"+++ b/{patch.filename}\n"
                f"{patch.patch_content}\n"
            )
        
        return diff_content
    
    def apply_patch_file_to_container(self, patch_file_path: Union[str, Path], 
                                     docker_executor, workdir: str, 
                                     include_test: bool = True, include_source: bool = True) -> Dict[str, any]:
        """Complete process of applying patch file to container"""
        # Read and parse patch file
        patch_content = self.read_patch_file(patch_file_path)
        patches = self.parse_unified_diff(patch_content)
        
        # Filter patches
        filtered_patches = self.filter_patches(patches, include_test, include_source)
        
        # Apply patches
        applied_files = self.apply_patches_to_container(filtered_patches, docker_executor, workdir)
        
        # Return results
        return {
            "total_files_num": len(filtered_patches),
            "applied_files_num": len(applied_files),
            "applied_files": applied_files,
            "patch_content": patch_content
        }
