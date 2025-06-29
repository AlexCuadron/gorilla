#!/usr/bin/env python3
"""
Simple script to generate CSV from tool scaling test results.
Uses the existing BFCL evaluation framework.
"""

import os
import re
import json
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm


def parse_folder_name(folder_name: str) -> Optional[Dict[str, any]]:
    """Parse folder name: model_number_of_tools_position_of_ground_truth_date"""
    pattern = r'^(.+?)_(\d+)_(\d+)_(\d{4}-\d{2}-\d{2})$'
    match = re.match(pattern, folder_name)
    
    if match:
        model_name, tools, position, date = match.groups()
        return {
            'model': model_name,
            'num_tools': int(tools),
            'position': int(position),
            'date': date
        }
    return None


def run_evaluation_for_folder(result_folder: Path, score_dir: Path) -> Optional[float]:
    """Run BFCL evaluation for a result folder and return accuracy."""
    try:
        # Extract model name from the folder structure
        model_subfolders = [d for d in result_folder.iterdir() if d.is_dir()]
        if not model_subfolders:
            print(f"No model subfolder found in {result_folder}")
            return None
        
        model_name = model_subfolders[0].name
        
        # Create score directory
        eval_score_dir = score_dir / result_folder.name
        eval_score_dir.mkdir(exist_ok=True)
        
        # Run evaluation
        cmd = [
            "python", "-m", "bfcl_eval", "evaluate",
            "--model", model_name,
            "--test-category", "simple", 
            "--result-dir", str(result_folder),
            "--score-dir", str(eval_score_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Look for score file
            score_file = eval_score_dir / model_name / "BFCL_v3_simple_score.json"
            if score_file.exists():
                with open(score_file, 'r') as f:
                    score_data = json.load(f)
                
                # Extract accuracy
                if isinstance(score_data, list) and len(score_data) > 0:
                    summary = score_data[0]
                    if 'accuracy' in summary:
                        return float(summary['accuracy'])
            
            print(f"Could not find accuracy in score file")
            return None
        else:
            print(f"Evaluation failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error evaluating {result_folder}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate CSV from tool scaling results")
    parser.add_argument("--result-dir", default="result", help="Directory containing result folders")
    parser.add_argument("--score-dir", default="score_temp", help="Temporary score directory")
    parser.add_argument("--output", default="evaluation_results_correct.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    score_dir = Path(args.score_dir)
    score_dir.mkdir(exist_ok=True)
    
    print("=== Tool Scaling CSV Generator ===")
    print(f"Result directory: {result_dir}")
    print(f"Output file: {args.output}")
    
    if not result_dir.exists():
        print(f"Result directory {result_dir} does not exist!")
        return
    
    results = []
    result_folders = [d for d in result_dir.iterdir() if d.is_dir()]
    print(f"Found {len(result_folders)} result folders")
    
    for folder in tqdm(result_folders, desc="Processing results"):
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            print(f"Could not parse folder name: {folder.name}")
            continue
        
        print(f"\nProcessing: {metadata['model']} | Tools: {metadata['num_tools']} | Position: {metadata['position']}")
        
        accuracy = run_evaluation_for_folder(folder, score_dir)
        
        if accuracy is not None:
            results.append({
                'model': metadata['model'],
                'number_of_tools': metadata['num_tools'],
                'ground_truth_pos': metadata['position'],
                'percentage_success': accuracy * 100  # Convert to percentage
            })
            print(f"✓ Success rate: {accuracy * 100:.2f}%")
        else:
            print(f"✗ Failed to get success rate")
    
    # Save results
    if results:
        # Sort results
        results.sort(key=lambda x: (x['model'], x['number_of_tools'], x['ground_truth_pos']))
        
        # Write to CSV
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['model', 'number_of_tools', 'ground_truth_pos', 'percentage_success']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n✅ Results saved to {args.output}")
        print(f"Total evaluations: {len(results)}")
        
        # Print summary
        models = set(r['model'] for r in results)
        print(f"Models evaluated: {sorted(models)}")
        
        avg_success = sum(r['percentage_success'] for r in results) / len(results)
        print(f"Average success rate: {avg_success:.2f}%")
        
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()