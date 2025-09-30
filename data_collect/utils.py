import re
import toml
from pathlib import Path

def load_config():
    """Load configuration file"""
    config_file = Path(__file__).parent / "config.toml"
    with open(config_file, 'r', encoding='utf-8') as f:
        return toml.load(f)

CONFIG = load_config()

def is_test_file(file_path: str) -> bool:
    """Check if file is a test file"""
    # Check if path contains test directory
    path_parts = file_path.lower().split('/')
    test_directories = CONFIG['release_collector']['test_directories']
    
    for part in path_parts:
        if part in test_directories:
            return True
    
    # Check if filename matches test file patterns
    file_name = Path(file_path).name
    test_patterns = CONFIG['release_collector']['test_file_patterns']
    
    return any(re.match(pattern, file_name) for pattern in test_patterns)