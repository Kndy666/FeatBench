import json
from typing import List, Dict

def process_entry(entry: Dict) -> List[Dict]:
    """Process a single raw entry, extract and convert to target format list"""
    processed = []
    repo = entry.get("repository")
    version = entry.get("release")
    
    # Process each enhanced_new_features
    for feature in entry.get("enhanced_new_features", []):
        # Process each pr_analyses
        for pr in feature.get("pr_analyses", []):
            # Extract basic information
            pr_number = pr.get("pr_number")
            base_commit = pr.get("base_commit", {}).get("sha", "")
            created_at = pr.get("base_commit", {}).get("date", "")
            detailed_desc = pr.get("detailed_description", "")
            
            # Extract organization name (first part of repo)
            org = repo.split("/")[0] if "/" in repo else repo
            
            # Generate instance_id
            instance_id = f"{repo.replace('/', '__')}-{pr_number}"
            
            # Get all file change records
            all_file_changes = pr.get("file_changes", [])
            
            # Get test_files list and extract corresponding changes
            test_file_names = pr.get("test_files", [])
            test_changes = [
                fc for fc in all_file_changes 
                if fc.get("filename") in test_file_names
            ]

            # Get non_test_files list and extract corresponding changes
            # non_test_file_names = pr.get("non_test_files", [])
            non_test_changes = [
                fc for fc in all_file_changes 
                if fc.get("filename") not in test_file_names
            ]
            
            # Build target format dictionary, directly save file change lists
            processed_item = {
                "repo": repo,
                "instance_id": instance_id,
                "base_commit": base_commit,
                "patch": non_test_changes,
                "test_patch": test_changes,
                "problem_statement": detailed_desc,
                "hints_text": "",
                "created_at": created_at,
                "version": version,
                "org": org,
                "number": int(pr_number) if pr_number else 0,
                "PASS_TO_PASS": "",
                "FAIL_TO_PASS": "",
                "test_files": test_file_names
            }
            processed.append(processed_item)
    
    return processed


def main(input_path: str, output_path: str):
    """Main function: read input JSON, process and write to output file"""
    # Read original JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process all entries
    all_processed = []
    for entry in data.get("results", []):
        processed = process_entry(entry)
        all_processed.extend(processed)
    
    # Deduplicate by instance_id, keep the last one
    dedup_map = {item["instance_id"]: item for item in all_processed}
    deduped = list(dedup_map.values())
    
    # Write processed results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed, generated {len(deduped)} records (deduplicated), saved to {output_path}")


if __name__ == "__main__":
    # Example usage (can be modified according to actual paths)
    input_json_path = "final_analysis_results.json"   # Input JSON file path
    output_json_path = "analysis_results.json" # Output result path
    main(input_json_path, output_json_path)