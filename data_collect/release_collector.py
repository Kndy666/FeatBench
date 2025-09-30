import requests
import time
import re
import json
import toml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm

# --- Configuration Loading ---
def load_config():
    """Load configuration file"""
    config_file = Path(__file__).parent / "config.toml"
    with open(config_file, 'r', encoding='utf-8') as f:
        return toml.load(f)

CONFIG = load_config()

# --- Configuration Area ---

TOKEN = CONFIG['common']['github_token']
HEADERS = {'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github.v3+json'}

# Crawling mode configuration
CRAWL_MODE = CONFIG['common']['crawl_mode']
CRAWL_JSON_FILE = Path(__file__).parent / CONFIG['common']['crawl_json_file']

# Filtering thresholds
MIN_STARS = CONFIG['release_collector']['min_stars_range']
RANK_START = CONFIG['release_collector']['rank_start']
RANK_END = CONFIG['release_collector']['rank_end']
MIN_RELEASES = CONFIG['release_collector']['min_releases']
MIN_RELEASE_BODY_LENGTH = CONFIG['release_collector']['min_release_body_length']
MIN_RELEASE_DATE = CONFIG['release_collector']['min_release_date']
EXCLUDED_TOPICS = set(CONFIG['release_collector']['excluded_topics'])

# Test case related configuration
TEST_DIRECTORIES = CONFIG['release_collector']['test_directories']
TEST_FILE_PATTERNS = CONFIG['release_collector']['test_file_patterns']
BOT_USERS = set(CONFIG['release_collector']['bot_users'])

# Cache file path
CACHE_FILE = Path(__file__).parent / CONFIG['common']['output_dir'] / CONFIG['release_collector']['cache_file']

# --- Data Class Definitions ---

@dataclass
class Release:
    """Represents a release version"""
    tag_name: str
    name: str
    body: str
    published_at: str
    target_commitish: str
    version_tuple: Tuple[int, ...]
    version_key: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Release':
        release = cls(**data)
        return release

@dataclass
class Repository:
    """Represents a repository and its release information"""
    full_name: str
    stargazers_count: int
    size: int
    topics: List[str]
    releases_count: int
    major_releases: List[Release]
    readme_content: str
    ci_configs: Dict[str, str]  # Added: CI/CD configuration file contents
    processed_at: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['major_releases'] = [release.to_dict() for release in self.major_releases]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Repository':
        repo = cls(**data)
        repo.major_releases = [Release.from_dict(release_data) for release_data in data.get("major_releases", [])]
        return repo

# --- Cache Management Functions ---

def load_processed_repos() -> Dict[str, Repository]:
    """Load processed repository information from JSON file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_repos = {}
                for repo_name, repo_data in data.items():
                    processed_repos[repo_name] = Repository.from_dict(repo_data)
                print(f"✅ Loaded {len(processed_repos)} processed repositories from cache file")
                return processed_repos
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Failed to load cache file: {e}, will restart processing")
            return {}
    else:
        print("📝 Cache file does not exist, will create new cache")
        return {}

def save_processed_repo(repository: Repository):
    """Save processing result of a single repository to JSON file."""
    # Load existing data
    processed_repos_dict = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                processed_repos_dict = json.load(f)
        except:
            pass
    
    # Add new repository data
    processed_repos_dict[repository.full_name] = repository.to_dict()
    
    # Save to file
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_repos_dict, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved processing result for repository {repository.full_name} to cache")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

# --- Core Function ---

def get_candidate_repos():
    """Get Python repositories within specified ranking range from GitHub API as candidate pool."""
    print(f"Getting Python repositories with Stars >= {MIN_STARS}, filtering ranking {RANK_START}-{RANK_END}...")
    
    API_URL = "https://api.github.com/search/repositories"
    PARAMS = {
        'q': f'language:python stars:>={MIN_STARS}',
        'sort': 'stars',
        'order': 'desc',
        'per_page': 100  # GitHub API max 100 per request
    }
    
    all_repos = []
    page = 1
    current_repo_count = 0
    
    try:
        while True:
            params_with_page = PARAMS.copy()
            params_with_page['page'] = page
            
            response = requests.get(API_URL, params=params_with_page, headers=HEADERS)
            response.raise_for_status()
            
            data = response.json()
            repos = data.get('items', [])
            
            if not repos:  # No more results
                break
                
            # Check ranking range for current page
            page_start_rank = current_repo_count + 1
            page_end_rank = current_repo_count + len(repos)
            
            print(f"✅ Retrieved page {page}, repository ranking {page_start_rank}-{page_end_rank}")
            
            # If starting rank of current page exceeds target end rank, stop fetching
            if page_start_rank > RANK_END:
                print(f"Exceeded target ranking range {RANK_END}, stopping fetch")
                break
            
            # Filter repositories within target ranking range
            for i, repo in enumerate(repos):
                repo_rank = current_repo_count + i + 1
                if RANK_START <= repo_rank <= RANK_END:
                    repo['rank'] = repo_rank  # Add ranking info
                    all_repos.append(repo)
            
            current_repo_count += len(repos)
            
            # If all repositories in target ranking range have been fetched, stop
            if page_end_rank >= RANK_END:
                print(f"Retrieved target ranking range {RANK_END}, stopping fetch")
                break
                
            # GitHub search API returns max 1000 results with pagination limit
            if current_repo_count >= data.get('total_count', 0) or page >= 10:
                break
                
            page += 1
            time.sleep(0.5)  # Avoid API limit
            
        print(f"✅ Total retrieved {len(all_repos)} repositories within ranking range {RANK_START}-{RANK_END}")
        return all_repos
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response.status_code == 403:
            print("API rate limit exceeded. Please use Token or wait and retry.")
        return []

def has_test_cases(repo_full_name: str) -> bool:
    """Check if repository contains test cases"""
    print(f"  > Checking if {repo_full_name} has test cases...")
    
    try:
        # 1. Check for test directories
        contents_url = f"https://api.github.com/repos/{repo_full_name}/contents"
        time.sleep(0.5)
        response = requests.get(contents_url, headers=HEADERS)
        response.raise_for_status()
        
        contents = response.json()
        
        # Check root directory for test-related directories
        has_test_directory = False
        test_directories_found = []
        for item in contents:
            if item.get('type') == 'dir':
                dir_name = item.get('name', '').lower()
                if any(test_dir in dir_name for test_dir in TEST_DIRECTORIES):
                    print(f"  > ✅ Found test directory: {item.get('name')}")
                    has_test_directory = True
                    test_directories_found.append(item.get('name'))
        
        # 2. Check root directory for test files
        for item in contents:
            if item.get('type') == 'file':
                file_name = item.get('name', '')
                if any(re.match(pattern, file_name) for pattern in TEST_FILE_PATTERNS):
                    print(f"  > ✅ Found test file: {file_name}")
                    return True
        
        # 3. Only if test directories found in root, recursively check test directory contents
        if has_test_directory:
            def check_directory_for_tests(repo_name, directory_path):
                """Recursively check if directory contains Python test files"""
                try:
                    dir_url = f"https://api.github.com/repos/{repo_name}/contents/{directory_path}"
                    time.sleep(0.5)
                    response = requests.get(dir_url, headers=HEADERS)
                    if response.status_code == 200:
                        contents = response.json()
                        if isinstance(contents, list):
                            # Check all files in directory at once
                            files = [item for item in contents if item.get('type') == 'file']
                            for item in files:
                                file_name = item.get('name', '')
                                # Check if Python file or test file
                                if file_name.endswith('.py') or any(re.match(pattern, file_name) for pattern in TEST_FILE_PATTERNS):
                                    print(f"  > ✅ Found Python file in test directory {directory_path}: {file_name}")
                                    return True
                            
                            # Then recursively check subdirectories
                            directories = [item for item in contents if item.get('type') == 'dir']
                            for dir_item in directories:
                                sub_dir_path = f"{directory_path}/{dir_item.get('name')}"
                                if check_directory_for_tests(repo_name, sub_dir_path):
                                    return True
                    return False
                except Exception as e:
                    print(f"  > ⚠️ Error checking directory {directory_path}: {e}")
                    return False
            
            # Recursively check each found test directory
            for test_dir in test_directories_found:
                if check_directory_for_tests(repo_full_name, test_dir):
                    return True

        print(f"  > ❌ No obvious test cases found")
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"  > ⚠️ Error checking test cases: {e}")
        return False
    except Exception as e:
        print(f"  > ⚠️ Exception checking test cases: {e}")
        return False

def is_valid_release(release_data: dict) -> bool:
    """Check if release is valid (not generated by bot and content is substantial and after specified date)"""
    # Check if generated by bot
    author_login = release_data.get('author', {}).get('login', '')
    if author_login in BOT_USERS:
        return False
    
    # Check release body length
    body = release_data.get('body', '') or ''  # Ensure body is not None
    if len(body.strip()) < MIN_RELEASE_BODY_LENGTH:
        return False
    
    # Check if publish time is after specified date
    published_at = release_data.get('published_at', '')
    if published_at:
        try:
            release_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            min_date = datetime.fromisoformat(MIN_RELEASE_DATE + 'T00:00:00+00:00')
            if release_date < min_date:
                return False
        except Exception:
            # If date parsing fails, skip this release
            return False
    else:
        # If no publish date, skip this release
        return False
    
    return True

def filter_by_metadata_and_releases(repos):
    """Preliminary filtering of repositories by macro indicators and release count via API."""
    print(f"Preliminary filtering by macro indicators and release count (ranking range {RANK_START}-{RANK_END})...")
    
    filtered_repos = []
    
    # Use tqdm to show filtering progress
    with tqdm(repos, desc="Filtering repositories", unit="repo") as pbar:
        for repo in pbar:
            repo_name = repo['full_name']
            repo_rank = repo.get('rank', 0)
            pbar.set_description(f"Checking: {repo_name} (rank#{repo_rank})")

            # 1. Check macro indicators (topics filtering)
            repo_topics = set(repo.get('topics', []))
            if repo_topics.intersection(EXCLUDED_TOPICS):
                pbar.write(f"  ❌ {repo_name} (#{repo_rank}): Contains excluded topics")
                continue

            # 2. Check if has test cases
            if not has_test_cases(repo_name):
                pbar.write(f"  ❌ {repo_name} (#{repo_rank}): No test cases")
                continue

            # 3. Check if has enough valid releases
            releases_url = f"https://api.github.com/repos/{repo_name}/releases"
            try:
                # Slight pause between checks to avoid API limit
                time.sleep(1)
                response = requests.get(releases_url, headers=HEADERS)
                response.raise_for_status()
                
                releases = response.json()
                # Filter valid releases
                valid_releases = [r for r in releases if is_valid_release(r)]
                
                if len(valid_releases) >= MIN_RELEASES:
                    repo['releases_count'] = len(valid_releases)
                    repo['releases_data'] = valid_releases  # Save filtered releases data
                    filtered_repos.append(repo)
                    pbar.write(f"  ✅ {repo_name} (#{repo_rank}): Passed initial screening! Stars: {repo['stargazers_count']}, Valid releases: {len(valid_releases)}")
                else:
                    pbar.write(f"  ❌ {repo_name} (#{repo_rank}): Insufficient valid releases, only {len(valid_releases)}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    pbar.write("\nAPI rate limit exceeded, initial screening terminated early.")
                    break
                pbar.write(f"  ❌ {repo_name} (#{repo_rank}): API Error: {e.response.status_code}")
    
    return filtered_repos

def extract_version_components(tag_name):
    """
    Extract version number components from tag name.
    Supports various formats like: v1.2.3, 1.2.3, 1-2-3, release-1.2.3, version.1.2.3, etc.
    Handles possible spaces in version numbers, like "v 1.2.3" or "1. 2. 3"
    
    Returns:
    - If version number successfully extracted, returns a version tuple (major, minor, patch, ...)
    - If unable to extract, returns None
    """
    # First clean input string, remove leading/trailing spaces
    tag_name = tag_name.strip()
    
    # 1. First try direct version pattern matching (not dependent on prefix judgment)
    # Match formats: number.number.number.number... or number-number-number... or number_number_number...
    direct_version_pattern = re.compile(r'(\d+)(?:\s*[.\-_]\s*(\d+))?(?:\s*[.\-_]\s*(\d+))?(?:\s*[.\-_]\s*(\d+))?')
    
    # First try matching version number from string start
    match = direct_version_pattern.match(tag_name)
    if match:
        version_tuple = tuple(int(group) for group in match.groups() if group is not None)
        if version_tuple:  # Ensure at least one version component
            return version_tuple
    
    # 2. If no match at start, try removing common prefixes then matching
    version_string = tag_name
    common_prefixes = ['version', 'release', 'ver', 'rel', 'v']  # Sorted by length, match longer first
    
    for prefix in common_prefixes:
        # Use regex for more precise prefix matching
        prefix_pattern = re.compile(rf'^{re.escape(prefix)}[.\-_\s]*', re.IGNORECASE)
        if prefix_pattern.match(tag_name):
            version_string = prefix_pattern.sub('', tag_name).strip()
            break
    
    # 3. Search for version pattern in processed string
    match = direct_version_pattern.match(version_string)
    if match:
        version_tuple = tuple(int(group) for group in match.groups() if group is not None)
        if version_tuple:
            return version_tuple
    
    # 4. Finally try searching for any version pattern anywhere in string
    match = direct_version_pattern.search(tag_name)
    if match:
        version_tuple = tuple(int(group) for group in match.groups() if group is not None)
        if version_tuple:
            return version_tuple
    
    return None

def get_major_releases(repo_full_name: str, releases_data, limit=5) -> List[Release]:
    """Get major version releases for repository (group by major version then take latest from each group)."""
    print(f"  > Getting major version releases for {repo_full_name}...")
    
    all_releases = releases_data
    print(f"  > Using {len(all_releases)} valid releases data already fetched")
    
    valid_releases = []
    
    for release in all_releases:
        tag_name = release.get('tag_name', '')
        
        # Use new version extraction function
        version_tuple = extract_version_components(tag_name)
        
        if version_tuple:
            # Skip pre-release versions (containing alpha, beta, rc, a, b identifiers)
            if re.search(r'(alpha|beta|rc|a\d+|b\d+)', tag_name.lower()):
                continue
            
            release_obj = Release(
                tag_name=tag_name,
                name=release.get('name', ''),
                body=release.get('body', ''),
                published_at=release.get('published_at', ''),
                target_commitish=release.get('target_commitish', ''),
                version_tuple=version_tuple,
                version_key='.'.join(str(v) for v in version_tuple),
            )
            valid_releases.append(release_obj)
    
    # Sort by version number, take latest few versions
    valid_releases.sort(key=lambda x: x.version_tuple, reverse=True)
    result = valid_releases[:limit]  # Take only first few versions
    
    print(f"  > Successfully got {len(result)} major version releases")
    if result:
        version_list = ', '.join([r.version_key for r in result])
        print(f"  > Selected versions: {version_list}")
    return result

def get_repository_readme(repo_full_name: str) -> str:
    """Get repository README content"""
    print(f"  > Getting README for {repo_full_name}...")
    
    try:
        # Get all files in repository root directory
        root_url = f"https://api.github.com/repos/{repo_full_name}/contents"
        time.sleep(0.5)  # Avoid API limit
        response = requests.get(root_url, headers=HEADERS)
        response.raise_for_status()
        
        contents = response.json()
        
        # Common README filename patterns
        readme_patterns = [r'^readme\.md$', r'^readme\.rst$', r'^readme\.txt$', r'^readme$']
        
        # Check locally if file list contains README
        for item in contents:
            if item.get('type') == 'file':
                file_name = item.get('name', '').lower()
                if any(re.match(pattern, file_name, re.IGNORECASE) for pattern in readme_patterns):
                    # Found README file, get content
                    download_url = item.get('download_url')
                    if download_url:
                        content_response = requests.get(download_url, headers=HEADERS)
                        content_response.raise_for_status()
                        readme_content = content_response.text
                        print(f"  > ✅ Successfully got README ({item.get('name')}), length: {len(readme_content)} characters")
                        return readme_content
        
        print(f"  > ❌ README file not found")
        return ""
    
    except Exception as e:
        print(f"  > ⚠️ Error getting README: {e}")
        return ""

def get_ci_configs(repo_full_name: str) -> Dict[str, str]:
    """Get list of CI/CD configuration files and download links for repository"""
    print(f"  > Getting CI/CD configuration file list for {repo_full_name}...")
    
    ci_configs = {}
    
    try:
        # Check if .github/workflows directory exists
        workflows_url = f"https://api.github.com/repos/{repo_full_name}/contents/.github/workflows"
        time.sleep(0.5)  # Avoid API limit
        response = requests.get(workflows_url, headers=HEADERS)
        
        # If directory exists, collect all YAML files info
        if response.status_code == 200:
            contents = response.json()
            
            for item in contents:
                if item.get('type') == 'file' and (item.get('name', '').endswith('.yml') or item.get('name', '').endswith('.yaml')):
                    file_name = item.get('name', '')
                    file_path = f".github/workflows/{file_name}"
                    download_url = item.get('download_url', '')
                    
                    if download_url:
                        ci_configs[file_path] = download_url
                        print(f"  > ✅ Found CI config: {file_path}")
        
        if ci_configs:
            print(f"  > ✅ Found {len(ci_configs)} CI configuration files total")
        else:
            print(f"  > ❌ No CI configuration files found")
        
        return ci_configs
    
    except Exception as e:
        print(f"  > ⚠️ Error getting CI config list: {e}")
        return {}

def process_single_repository(repo: Dict, use_cache: bool = True) -> Repository:
    """Process single repository, get its details"""
    repo_name = repo['full_name']
    
    # Get major version releases, using limit from config
    major_releases = get_major_releases(
        repo_name, 
        releases_data=repo.get('releases_data'), 
        limit=CONFIG['release_collector']['default_release_limit']
    )
    if not major_releases:
        raise ValueError(f"Unable to get major version releases")
        
    # Get README content
    readme_content = get_repository_readme(repo_name)

    # Get CI/CD configurations
    ci_configs = get_ci_configs(repo_name)
    
    # Create Repository object
    repository = Repository(
        full_name=repo_name,
        stargazers_count=repo['stargazers_count'],
        size=repo['size'],
        topics=repo.get('topics', []),
        releases_count=repo['releases_count'],
        major_releases=major_releases,
        readme_content=readme_content,
        ci_configs=ci_configs,
        processed_at=time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save to cache
    if use_cache:
        save_processed_repo(repository)
    
    return repository

def get_specified_repos():
    """Get specified repository list from crawl.json file"""
    print(f"Getting repository list from specified file: {CRAWL_JSON_FILE}")
    
    if not CRAWL_JSON_FILE.exists():
        print(f"❌ Specified repository file does not exist: {CRAWL_JSON_FILE}")
        return []
    
    try:
        with open(CRAWL_JSON_FILE, 'r', encoding='utf-8') as f:
            crawl_data = json.load(f)
        
        # Collect repositories from all categories
        all_repos = []
        for category, repos in crawl_data.items():
            print(f"✅ Loaded category '{category}': {len(repos)} repositories")
            for repo_name in repos:
                all_repos.append(repo_name)
        
        print(f"✅ Total loaded {len(all_repos)} specified repositories")
        
        # Get details for each repository
        detailed_repos = []
        with tqdm(all_repos, desc="Getting repository info", unit="repo") as pbar:
            for repo_name in pbar:
                pbar.set_description(f"Getting: {repo_name}")
                try:
                    repo_info = get_repository_info(repo_name)
                    if repo_info:
                        detailed_repos.append(repo_info)
                        pbar.write(f"  ✅ {repo_name}: Stars {repo_info['stargazers_count']}")
                    else:
                        pbar.write(f"  ❌ {repo_name}: Failed to get info")
                except Exception as e:
                    pbar.write(f"  ❌ {repo_name}: {str(e)}")
                    continue
                
                time.sleep(0.5)  # Avoid API limit
        
        print(f"✅ Successfully got details for {len(detailed_repos)} repositories")
        return detailed_repos
        
    except Exception as e:
        print(f"❌ Failed to read specified repository file: {e}")
        return []

def get_repository_info(repo_name: str) -> Dict:
    """Get detailed information for single repository"""
    try:
        repo_url = f"https://api.github.com/repos/{repo_name}"
        response = requests.get(repo_url, headers=HEADERS)
        response.raise_for_status()
        
        repo_data = response.json()
        
        # Return data in same format as get_candidate_repos
        return {
            'full_name': repo_data['full_name'],
            'stargazers_count': repo_data['stargazers_count'],
            'size': repo_data['size'],
            'topics': repo_data.get('topics', []),
            'language': repo_data.get('language', ''),
            'archived': repo_data.get('archived', False),
            'disabled': repo_data.get('disabled', False),
            'fork': repo_data.get('fork', False),
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  ⚠️ Repository does not exist: {repo_name}")
        else:
            print(f"  ⚠️ Failed to get repository info: {repo_name} - {e}")
        return None
    except Exception as e:
        print(f"  ⚠️ Exception getting repository info: {repo_name} - {e}")
        return None

def get_repositories_to_process(use_cache: bool = True, crawl_mode: str = None) -> Tuple[List[Dict], Dict[str, Repository]]:
    """Get list of repositories to process and processed repositories"""
    # Load processed repository cache
    processed_repos = load_processed_repos() if use_cache else {}
    
    # Determine crawl mode
    mode = crawl_mode or CRAWL_MODE
    
    # Get candidate repositories based on mode
    if mode == "specified":
        print("🎯 Using specified repository mode")
        candidate_repos = get_specified_repos()
    else:
        print("⭐ Using star count filtering mode")
        candidate_repos = get_candidate_repos()
    
    if not candidate_repos:
        return [], processed_repos

    # Filter out already processed repositories
    if processed_repos:
        unprocessed_repos = [repo for repo in candidate_repos if repo['full_name'] not in processed_repos]
        candidate_repos = unprocessed_repos

    # Preliminary filtering
    pre_filtered_repos = filter_by_metadata_and_releases(candidate_repos)
    
    return pre_filtered_repos, processed_repos