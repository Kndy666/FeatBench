import re
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


class TestStatus(Enum):
    """Test status enumeration"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class PytestResultParser:
    """
    Tool class for parsing pytest output results
    Supports parsing pytest -q -rA --tb=np format output
    """
    
    def __init__(self, output: str):
        """
        Initialize parser
        
        Args:
            output: pytest output string
        """
        self.output = output
        self.test_results: Dict[str, TestStatus] = {}
        self._parse_output()
    
    def _clean_ansi_codes(self, text: str) -> str:
        """Clean ANSI escape codes"""
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)
    
    def _parse_output(self):
        """Parse pytest output"""
        clean_output = self._clean_ansi_codes(self.output)
        
        # Find "short test summary info" section
        summary_start = clean_output.find("short test summary info")
        if summary_start == -1:
            # If summary section not found, all tests may have passed, try parsing from full output
            self._parse_from_full_output(clean_output)
            return
        
        # Extract content after summary section
        summary_section = clean_output[summary_start:]
        
        # Split by lines and parse each line
        lines = summary_section.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse test result line
            self._parse_test_line(line)
    
    def _parse_from_full_output(self, clean_output: str):
        """Parse test results from full output (when no summary section)"""
        lines = clean_output.split('\n')
        for line in lines:
            line = line.strip()
            # Find lines containing test results
            if any(status.value in line for status in TestStatus):
                self._parse_test_line(line)
    
    def _parse_test_line(self, line: str):
        """Parse single line test result"""
        # Match format: STATUS test_file.py::TestClass::test_method[params] - error_message
        # Or: STATUS test_file.py::test_function
        pattern = r'^(PASSED|FAILED|SKIPPED|ERROR)\s+(.+?)(?:\s-\s.*)?$'
        
        match = re.match(pattern, line)
        if match:
            status_str = match.group(1)
            test_path = match.group(2).strip()
            
            try:
                status = TestStatus(status_str)
                self.test_results[test_path] = status
            except ValueError:
                # If status cannot be recognized, mark as UNKNOWN
                self.test_results[test_path] = TestStatus.UNKNOWN
    
    def _get_base_test_name(self, test_path: str) -> str:
        """
        Get base test name (remove parametrized part)
        
        Args:
            test_path: Complete test path
            
        Returns:
            Base test name
        """
        # Remove parametrized part [param1-param2-param3]
        base_name = test_path.split('[')[0] if '[' in test_path else test_path
        return base_name
    
    def _aggregate_parametrized_results(self, test_results: Dict[str, TestStatus]) -> TestStatus:
        """
        Aggregate parametrized test results
        Rules:
        - If all results are passed or contain skipped (at least one passed) return passed
        - If any failed, errored, unknown, return failed
        
        Args:
            test_results: All parametrized results for same base test name
            
        Returns:
            Aggregated test status
        """
        if not test_results:
            return TestStatus.UNKNOWN
        
        statuses = list(test_results.values())
        
        # If any failed, error or unknown, return failed
        if any(status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.UNKNOWN] for status in statuses):
            return TestStatus.FAILED
        
        # If all are passed or skipped, and at least one passed, return passed
        if all(status in [TestStatus.PASSED, TestStatus.SKIPPED] for status in statuses):
            if any(status == TestStatus.PASSED for status in statuses):
                return TestStatus.PASSED
            else:
                # All are skipped
                return TestStatus.SKIPPED
        
        # Other cases return unknown
        return TestStatus.UNKNOWN
    
    def get_test_status(self, test_pattern: str) -> Optional[TestStatus]:
        """
        Get status of specified test
        For parametrized tests, automatically aggregates all parameter results
        
        Args:
            test_pattern: Test pattern, like "test_api_jws.py::TestJWS::test_encode_with_jwk"
        
        Returns:
            Test status enum, returns None if not found
        """
        # First check for exact match
        if test_pattern in self.test_results:
            return self.test_results[test_pattern]
        
        # Check if it's base test name (may have parametrized variants)
        base_name = self._get_base_test_name(test_pattern)
        parametrized_results = {}
        
        for test_path, status in self.test_results.items():
            if self._get_base_test_name(test_path) == base_name:
                parametrized_results[test_path] = status
        
        if parametrized_results:
            return self._aggregate_parametrized_results(parametrized_results)
        
        return None
    
    def query_tests(self, test_patterns: List[str]) -> Dict[str, TestStatus]:
        """
        Query status of multiple tests
        
        Args:
            test_patterns: Test pattern list
        
        Returns:
            Mapping dict from test pattern to status
        """
        results = {}
        for pattern in test_patterns:
            status = self.get_test_status(pattern)
            results[pattern] = status if status else TestStatus.UNKNOWN
        return results
    
    def find_tests_by_pattern(self, pattern: str) -> Dict[str, TestStatus]:
        """
        Find matching tests by pattern
        
        Args:
            pattern: Regex pattern or simple string match
        
        Returns:
            Matching tests and their statuses
        """
        results = {}
        
        # If pattern contains regex special characters, use regex match
        if any(char in pattern for char in r'.*+?^${}[]|()\\/'):
            regex = re.compile(pattern)
            for test_path, status in self.test_results.items():
                if regex.search(test_path):
                    results[test_path] = status
        else:
            # Simple string match
            for test_path, status in self.test_results.items():
                if pattern in test_path:
                    results[test_path] = status
        
        return results
    
    def find_tests_by_base_name(self, base_pattern: str) -> Dict[str, TestStatus]:
        """
        Find all parametrized tests by base test name
        
        Args:
            base_pattern: Base test name, like "test_max_chat_history"
        
        Returns:
            All matching parametrized tests and their statuses
        """
        results = {}
        for test_path, status in self.test_results.items():
            # Extract base test name (remove parametrized part)
            base_test = self._get_base_test_name(test_path)
            if base_pattern in base_test:
                results[test_path] = status
        return results
    
    def get_all_results(self) -> Dict[str, TestStatus]:
        """Get all parsed test results"""
        return self.test_results.copy()
    
    def filter_tests_by_status(self, expected_statuses: Optional[List[TestStatus]] = None) -> Set[str]:
        """
        Filter test items that match expected status (aggregated parametrized results).

        Args:
            expected_statuses: Expected status list (e.g. [TestStatus.PASSED])

        Returns:
            Set of base test paths that match and have aggregated status in expected_statuses
        """
        if expected_statuses is None or not expected_statuses:
            expected_statuses = [TestStatus.PASSED]

        matched: Set[str] = set()
        # Group by base test name
        base_test_groups = defaultdict(dict)
        for test_path, status in self.test_results.items():
            base_name = self._get_base_test_name(test_path)
            base_test_groups[base_name][test_path] = status

        # Aggregate and filter
        for base_name, group_results in base_test_groups.items():
            aggregated = self._aggregate_parametrized_results(group_results)
            if aggregated in expected_statuses:
                matched.add(base_name)

        return matched
    
    def get_summary(self) -> Dict[TestStatus, int]:
        """
        Get test result statistics
        
        Returns:
            Mapping from status to count
        """
        summary = {status: 0 for status in TestStatus}
        for status in self.test_results.values():
            summary[status] += 1
        return summary
    
    def get_aggregated_summary(self) -> Dict[TestStatus, int]:
        """
        Get aggregated test result statistics (parametrized tests aggregated by base name)
        
        Returns:
            Mapping from status to count
        """
        # Group by base test name
        base_test_groups = defaultdict(dict)
        for test_path, status in self.test_results.items():
            base_name = self._get_base_test_name(test_path)
            base_test_groups[base_name][test_path] = status
        
        # Aggregate results for each base test
        aggregated_results = {}
        for base_name, test_results in base_test_groups.items():
            aggregated_results[base_name] = self._aggregate_parametrized_results(test_results)
        
        # Count aggregated results
        summary = {status: 0 for status in TestStatus}
        for status in aggregated_results.values():
            summary[status] += 1
        
        return summary
    
    def check_all_tests_status(self, test_patterns: List[str], expected_statuses: List[TestStatus] = None) -> Tuple[bool, Dict[str, TestStatus]]:
        """
        Check if specified tests all meet expected status
        
        Args:
            test_patterns: List of test patterns to check
            expected_statuses: Expected status list, defaults to [PASSED]
        
        Returns:
            (Whether all meet expectations, actual status dict)
        """
        if expected_statuses is None:
            expected_statuses = [TestStatus.PASSED]
        
        results = self.query_tests(test_patterns)
        all_expected = all(status in expected_statuses for status in results.values())
        
        return all_expected, results