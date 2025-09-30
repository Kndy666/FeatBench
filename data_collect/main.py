import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import sys
import toml
from tqdm import tqdm

from release_collector import (
    Repository,
    load_processed_repos,
    get_repositories_to_process,
    process_single_repository,
)
from release_analyzer import analyze_repository_releases, ReleaseAnalysis, load_analysis_cache
from pr_analyzer import enhance_release_analysis_with_pr_details, load_pr_analysis_cache

# --- Configuration Loading ---
def load_config():
    """Load configuration file"""
    config_file = Path(__file__).parent / "config.toml"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to load configuration file: {e}")
        sys.exit(1)

CONFIG = load_config()
OUTPUT_DIR = Path(__file__).parent / CONFIG['common']['output_dir']
FINAL_RESULTS_FILE = OUTPUT_DIR / CONFIG['main']['final_results_file']

def setup_output_directory():
    """Create output directory"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")

def collect_repositories(use_cache: bool = True, crawl_mode: str = None) -> List[Repository]:
    """Collect and process repositories"""
    print("\n🔍 === Step 1: Collect Repository Information ===")
    
    # Get list of repositories to process and processed repositories
    pre_filtered_repos, processed_repos = get_repositories_to_process(use_cache, crawl_mode)
    
    if not pre_filtered_repos:
        print("❌ No repositories passed initial filtering")
        # If no new repositories passed filtering, but there are processed repositories, return processed repositories
        if processed_repos:
            print(f"📂 Returning {len(processed_repos)} processed repositories")
            return list(processed_repos.values())
        return []
        
    print(f"✅ {len(pre_filtered_repos)} repositories passed initial filtering")
    
    # Process each repository
    final_repositories = []
    
    # First add processed repositories
    if processed_repos:
        final_repositories.extend(processed_repos.values())
        print(f"📂 Loaded {len(processed_repos)} processed repositories")
    
    # Use tqdm to show processing progress
    with tqdm(pre_filtered_repos, desc="Processing repositories", unit="repo") as pbar:
        for repo in pbar:
            repo_name = repo['full_name']
            pbar.set_description(f"Processing: {repo_name}")
            
            try:
                repository = process_single_repository(repo, use_cache)
                final_repositories.append(repository)
                pbar.write(f"  ✅ {repo_name}: Processing completed")
            except Exception as e:
                pbar.write(f"  ❌ {repo_name}: {str(e)}")
                continue
    
    print(f"\n✅ Collection phase completed, processed {len(final_repositories)} repositories")
    return final_repositories

def analyze_releases(repositories: List[Repository]) -> List[ReleaseAnalysis]:
    """Analyze releases for all repositories"""
    print("\n📊 === Step 2: Analyze Release Features ===")
    
    all_analyses = []
    
    # Use tqdm to show analysis progress
    with tqdm(repositories, desc="Analyzing repositories", unit="repo") as pbar:
        for repository in pbar:
            pbar.set_description(f"Analyzing: {repository.full_name}")
            analyses = analyze_repository_releases(repository)
            all_analyses.extend(analyses)
            
            # Count analysis results for current repository
            total_new_features = sum(len(a.new_features) for a in analyses)
            total_improvements = sum(len(a.improvements) for a in analyses)
            total_bug_fixes = sum(len(a.bug_fixes) for a in analyses)
            
            pbar.write(f"  ✅ {repository.full_name}: New features({total_new_features}) Improvements({total_improvements}) Fixes({total_bug_fixes})")
    
    # Count overall results
    total_new_features = sum(len(a.new_features) for a in all_analyses)
    total_improvements = sum(len(a.improvements) for a in all_analyses)
    total_bug_fixes = sum(len(a.bug_fixes) for a in all_analyses)
    
    print(f"\n✅ Release analysis completed!")
    print(f"  - Analyzed {len(all_analyses)} releases")
    print(f"  - New features: {total_new_features}")
    print(f"  - Improvements: {total_improvements}")
    print(f"  - Bug fixes: {total_bug_fixes}")
    
    return all_analyses

def enhance_with_pr_analysis(release_analyses: List[ReleaseAnalysis]) -> List[Dict]:
    """Enhance PR analysis"""
    print("\n🔧 === Step 3: Enhance PR Detailed Analysis ===")
    
    enhanced_results = []
    
    # Use tqdm to show PR analysis progress
    with tqdm(release_analyses, desc="Analyzing PRs", unit="release") as pbar:
        for analysis in pbar:
            pbar.set_description(f"Analyzing: {analysis.repo_name}-{analysis.tag_name}")
            
            # Only perform detailed PR analysis for new features
            enhanced_features = enhance_release_analysis_with_pr_details(analysis)
            
            if enhanced_features:
                result = {
                    'repository': analysis.repo_name,
                    'release': analysis.tag_name,
                    'analyzed_at': analysis.analyzed_at,
                    'enhanced_new_features': [ef.to_dict() for ef in enhanced_features],
                    'original_analysis': analysis.to_dict()
                }
                enhanced_results.append(result)
                pbar.write(f"  ✅ {analysis.repo_name}-{analysis.tag_name}: Analyzed {len(enhanced_features)} PRs")
            else:
                pbar.write(f"  ⚠️ {analysis.repo_name}-{analysis.tag_name}: No analyzable new features")
    
    print(f"\n✅ PR detailed analysis completed, analyzed {len(enhanced_results)} releases")
    return enhanced_results

def save_final_results(enhanced_results: List[Dict]):
    """Save final results"""
    print("\n💾 === Step 4: Save Final Results ===")
    
    final_output = {
        'metadata': {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_repositories': len(set(r['repository'] for r in enhanced_results)),
            'total_releases': len(enhanced_results),
            'total_enhanced_features': sum(len(r['enhanced_new_features']) for r in enhanced_results)
        },
        'results': enhanced_results
    }
    
    try:
        with open(FINAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"✅ Final results saved to: {FINAL_RESULTS_FILE}")
        
        # Print results summary
        print(f"\n📈 === Final Statistics ===")
        print(f"  - Analyzed repositories: {final_output['metadata']['total_repositories']}")
        print(f"  - Analyzed releases: {final_output['metadata']['total_releases']}")
        print(f"  - Enhanced features: {final_output['metadata']['total_enhanced_features']}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def print_sample_results(enhanced_results: List[Dict], limit: int = 5):
    """Print sample results"""
    # If no limit specified, use default from config
    if limit is None:
        limit = CONFIG['main']['sample_results_limit']
        
    print(f"\n🎯 === Sample Results Preview (First {limit}) ===")
    
    for i, result in enumerate(enhanced_results[:limit]):
        print(f"\n--- Sample {i+1}: {result['repository']} - {result['release']} ---")
        
        enhanced_features = result['enhanced_new_features']
        for j, feature in enumerate(enhanced_features[:2]):  # Show only first 2 features per release
            print(f"  Feature {j+1}: {feature['description'][:100]}...")
            if feature['pr_analyses']:
                pr_count = len(feature['pr_analyses'])
                print(f"    - Associated PRs: {pr_count}")
                print(f"    - Detailed description: {feature['feature_detailed_description'][:150]}...")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GitHub repository release and PR analysis tool')
    parser.add_argument('--no-cache', action='store_true',
                       help='Do not use cache, reprocess all data')
    parser.add_argument('--collect-only', action='store_true',
                       help='Only perform repository collection, skip subsequent analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only perform release analysis, skip repository collection')
    parser.add_argument('--enhance-only', action='store_true',
                       help='Only perform PR enhancement analysis, skip previous steps')
    parser.add_argument('--crawl-mode', choices=['stars', 'specified'], default='specified',
                       help='Choose crawl mode: stars (filter by star count) or specified (use specified repository list)')
    
    args = parser.parse_args()
    
    print("🚀 Starting GitHub repository analysis process")
    print("=" * 50)
    
    # Setup output directory
    setup_output_directory()
    
    use_cache = not args.no_cache
    enhanced_results = []
    
    try:
        if not args.analyze_only and not args.enhance_only:
            # Step 1: Collect repositories
            repositories = collect_repositories(use_cache=use_cache, crawl_mode=args.crawl_mode)
            
            if not repositories:
                print("❌ No valid repositories collected, program ends")
                return
                
            if args.collect_only:
                print(f"✅ Collection-only mode completed, collected {len(repositories)} repositories")
                return
        else:
            # Load repository data from cache
            processed_repos = load_processed_repos()
            repositories = list(processed_repos.values())
            print(f"📂 Loaded {len(repositories)} repositories from cache")

        if not args.enhance_only:
            # Step 2: Analyze releases
            release_analyses = analyze_releases(repositories)
            
            if not release_analyses:
                print("❌ No valid releases analyzed, program ends")
                return
                
            if args.analyze_only:
                print("✅ Analysis-only mode completed")
                return
        else:
            cached_analyses = load_analysis_cache()
            release_analyses = list(cached_analyses.values())
            print(f"📂 Loaded {len(release_analyses)} release analyses from cache")
        
        # Step 3: Enhance PR analysis
        cached_pr_analysis = load_pr_analysis_cache()
        pr_analysis = list(cached_pr_analysis.values())
        print(f"📂 Loaded {len(pr_analysis)} PR analyses from cache")

        enhanced_results = enhance_with_pr_analysis(release_analyses)

        if enhanced_results:
            save_final_results(enhanced_results)
            print_sample_results(enhanced_results)
        
        print(f"\n🎉 Complete analysis process finished!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Program execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()