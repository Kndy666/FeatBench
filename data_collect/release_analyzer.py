import json
import time
import toml
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import openai
from pathlib import Path
from tqdm import tqdm

# --- Configuration Loading ---
def load_config():
    """Load configuration file"""
    config_file = Path(__file__).parent / "config.toml"
    with open(config_file, 'r', encoding='utf-8') as f:
        return toml.load(f)

CONFIG = load_config()

# --- Configuration Area ---
OPENAI_API_KEY = CONFIG['common']['openai_api_key']
OPENAI_MODEL = CONFIG['common']['openai_model']

# Cache file
ANALYSIS_CACHE_FILE = Path(__file__).parent / CONFIG['common']['output_dir'] / CONFIG['release_analyzer']['analysis_cache_file']

# --- Data Class Definitions ---

@dataclass
class FeatureAnalysis:
    """Represents a feature analysis result"""
    feature_type: str  # 'new_feature', 'improvement', 'bug_fix', 'other'
    description: str
    pr_links: List[str]  # Related PR links
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureAnalysis':
        return cls(**data)

@dataclass
class ReleaseAnalysis:
    """Represents a release analysis result"""
    tag_name: str
    repo_name: str
    new_features: List[FeatureAnalysis]
    improvements: List[FeatureAnalysis]
    bug_fixes: List[FeatureAnalysis]
    other_changes: List[FeatureAnalysis]
    processed_body: str  # Body with PR links processed
    analyzed_at: str
    
    def to_dict(self) -> Dict:
        return {
            'tag_name': self.tag_name,
            'repo_name': self.repo_name,
            'new_features': [f.to_dict() for f in self.new_features],
            'improvements': [f.to_dict() for f in self.improvements],
            'bug_fixes': [f.to_dict() for f in self.bug_fixes],
            'other_changes': [f.to_dict() for f in self.other_changes],
            'processed_body': self.processed_body,
            'analyzed_at': self.analyzed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReleaseAnalysis':
        return cls(
            tag_name=data['tag_name'],
            repo_name=data['repo_name'],
            new_features=[FeatureAnalysis.from_dict(f) for f in data.get('new_features', [])],
            improvements=[FeatureAnalysis.from_dict(f) for f in data.get('improvements', [])],
            bug_fixes=[FeatureAnalysis.from_dict(f) for f in data.get('bug_fixes', [])],
            other_changes=[FeatureAnalysis.from_dict(f) for f in data.get('other_changes', [])],
            processed_body=data.get('processed_body', ''),
            analyzed_at=data.get('analyzed_at', '')
        )

# --- Cache Management ---

def load_analysis_cache() -> Dict[str, ReleaseAnalysis]:
    """Load analysis cache"""
    if ANALYSIS_CACHE_FILE.exists():
        try:
            with open(ANALYSIS_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cache = {}
                for key, analysis_data in data.items():
                    cache[key] = ReleaseAnalysis.from_dict(analysis_data)
                print(f"✅ Loaded {len(cache)} release analysis results from cache")
                return cache
        except Exception as e:
            print(f"⚠️ Failed to load analysis cache: {e}")
            return {}
    return {}

def save_analysis_to_cache(analysis: ReleaseAnalysis):
    """Save analysis result to cache"""
    cache = {}
    if ANALYSIS_CACHE_FILE.exists():
        try:
            with open(ANALYSIS_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except:
            pass
    
    cache_key = f"{analysis.repo_name}#{analysis.tag_name}"
    cache[cache_key] = analysis.to_dict()
    
    try:
        with open(ANALYSIS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved analysis result for {cache_key} to cache")
    except Exception as e:
        print(f"⚠️ Failed to save analysis cache: {e}")

# --- LLM Analysis ---

def analyze_release_with_llm(release_body: str, tag_name: str, repo_readme: str = "") -> Dict[str, List[Dict]]:
    """Use LLM to analyze feature changes and PR links in release body"""
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url="")
    
    # Build prompt with README context
    readme_context = ""
    if repo_readme.strip():
        # Read parameters from config
        max_readme_length = CONFIG['release_analyzer']['max_readme_length']
        truncation_suffix = CONFIG['release_analyzer']['readme_truncation_suffix']
        
        # Truncate README to avoid overly long prompt
        readme_excerpt = repo_readme[:max_readme_length]
        if len(repo_readme) > max_readme_length:
            readme_excerpt += truncation_suffix
        readme_context = f"""
Repository Context (README):
{readme_excerpt}

---
"""
    
    prompt = f"""
{readme_context}Analyze the following software release notes and categorize the changes into: new_features, improvements, bug_fixes, and other_changes.
For each change, extract any PR references (like #123, PR456, pull #789, etc.) mentioned in the text.

Release version: {tag_name}
Release notes:
{release_body}

Guidelines:
1. new_features: Brand new functionality, commands, rules, or capabilities
2. improvements: Enhancements to existing features, optimizations, performance improvements
3. bug_fixes: Bug fixes, error handling, crash fixes
4. other_changes: Documentation updates, dependency updates, refactoring (only if significant)
5. Extract PR numbers from various formats: #123, PR #456, pull 789, (#101), etc.
6. Only include PR numbers that are explicitly mentioned with the change
7. Ignore trivial changes like version bumps unless they're part of larger features
8. Use the repository context to better understand the project's domain and categorize changes more accurately

Return the result in JSON format:
{
    "new_features": [
        {
            "description": "Brief description of the new feature",
            "pr_ids": ["123", "456"]
        }
    ],
    "improvements": [
        {
            "description": "Brief description of the improvement", 
            "pr_ids": ["789"]
        }
    ],
    "bug_fixes": [
        {
            "description": "Brief description of the bug fix",
            "pr_ids": ["101"]
        }
    ],
    "other_changes": [
        {
            "description": "Brief description of other changes",
            "pr_ids": []
        }
    ]
}
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a software development expert who specializes in analyzing release notes and categorizing software changes."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content is None:
            print("⚠️ LLM returned empty content")
            return {"new_features": [], "improvements": [], "bug_fixes": [], "other_changes": []}
        result = json.loads(content)
        return result
            
    except Exception as e:
        print(f"⚠️ LLM analysis failed: {e}")
        return {"new_features": [], "improvements": [], "bug_fixes": [], "other_changes": []}

# --- Main Function ---

def analyze_release(release, repo_name: str, repo_readme: str = "", use_cache: bool = True) -> Optional[ReleaseAnalysis]:
    """Analyze a single release"""
    cache_key = f"{repo_name}#{release.tag_name}"
    
    # Check cache
    if use_cache:
        cache = load_analysis_cache()
        if cache_key in cache:
            print(f"  > 🔄 Loading analysis result for {release.tag_name} from cache")
            return cache[cache_key]
    
    print(f"  > 🔍 Analyzing release for {release.tag_name}...")
    
    # LLM analysis, pass README content
    llm_result = analyze_release_with_llm(release.body, release.tag_name, repo_readme)
    
    # Convert to FeatureAnalysis objects
    def convert_to_feature_analysis(items: List[Dict], feature_type: str) -> List[FeatureAnalysis]:
        features = []
        for item in items:
            pr_links = []
            # Get PR IDs from LLM result and convert to full links
            if 'pr_ids' in item:
                for pr_id in item['pr_ids']:
                    pr_links.append(f"https://github.com/{repo_name}/pull/{pr_id}")
            
            features.append(FeatureAnalysis(
                feature_type=feature_type,
                description=item.get('description', ''),
                pr_links=pr_links
            ))
        return features
    
    analysis = ReleaseAnalysis(
        tag_name=release.tag_name,
        repo_name=repo_name,
        new_features=convert_to_feature_analysis(llm_result.get('new_features', []), 'new_feature'),
        improvements=convert_to_feature_analysis(llm_result.get('improvements', []), 'improvement'),
        bug_fixes=convert_to_feature_analysis(llm_result.get('bug_fixes', []), 'bug_fix'),
        other_changes=convert_to_feature_analysis(llm_result.get('other_changes', []), 'other'),
        processed_body=release.body,
        analyzed_at=time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save to cache
    if use_cache:
        save_analysis_to_cache(analysis)
    
    return analysis

def analyze_repository_releases(repository) -> List[ReleaseAnalysis]:
    """Analyze all major releases of the repository"""
    print(f"--- Starting analysis of release features for repository {repository.full_name} ---")
    
    analyses = []
    
    # Use tqdm to show analysis progress
    with tqdm(repository.major_releases, desc=f"Analyzing {repository.full_name}", unit="release") as pbar:
        for release in pbar:
            pbar.set_description(f"Analyzing: {release.tag_name}")
            
            # Pass README content to analysis function
            analysis = analyze_release(release, repository.full_name, repository.readme_content)
            if analysis:
                analyses.append(analysis)
                # Show analysis result summary
                new_features_count = len(analysis.new_features)
                improvements_count = len(analysis.improvements)
                bug_fixes_count = len(analysis.bug_fixes)
                pbar.write(f"    ✅ {release.tag_name}: New features({new_features_count}) Improvements({improvements_count}) Fixes({bug_fixes_count})")
            
            # Avoid API rate limit
            time.sleep(1)
    
    return analyses