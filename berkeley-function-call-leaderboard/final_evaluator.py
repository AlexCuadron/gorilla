#!/usr/bin/env python3
"""
Final evaluation script for tool scaling test results.

This script properly evaluates the results using the existing BFCL evaluation framework.
"""

import os
import re
import json
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm


class FinalEvaluator:
    def __init__(self, result_dir: str = "result", score_dir: str = "score"):
        self.result_dir = Path(result_dir)
        self.score_dir = Path(score_dir)
        self.score_dir.mkdir(exist_ok=True)
        
    def parse_folder_name(self, folder_name: str) -> Optional[Dict[str, any]]:
        """Parse folder name to extract metadata."""
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
    
    def find_available_test_files(self) -> Dict[Tuple[int, int], str]:
        """Find all available test files and map them to (tools, position) tuples."""
        data_dir = Path("bfcl_eval/data")
        test_files = {}
        
        for file_path in data_dir.glob("BFCL_v3_simple_tools_*_pos_*.json"):
            # Extract tools and position from filename
            match = re.search(r'tools_(\d+)_pos_(\d+)', file_path.name)
            if match:
                tools, position = int(match.group(1)), int(match.group(2))
                test_files[(tools, position)] = str(file_path)
        
        return test_files
    
    def run_bfcl_evaluation(self, metadata: Dict, test_file: str) -> Optional[float]:
        """Run BFCL evaluation for a specific result."""
        try:
            # Create a unique score directory for this evaluation
            eval_score_dir = self.score_dir / f"{metadata['model']}_{metadata['num_tools']}_{metadata['position']}"
            eval_score_dir.mkdir(exist_ok=True)
            
            # Find the result directory
            result_folder = self.result_dir / f"{metadata['model']}_{metadata['num_tools']}_{metadata['position']}_{metadata['date']}"
            
            if not result_folder.exists():
                print(f"Warning: Result folder not found: {result_folder}")
                return None
            
            # Run the evaluation using the BFCL command line interface
            cmd = [
                "python", "-m", "bfcl_eval", "evaluate",
                "--model", metadata['model'],
                "--test-category", "simple",
                "--result-dir", str(result_folder),
                "--score-dir", str(eval_score_dir),
                "--custom-path", test_file
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Look for the score file
                score_file = eval_score_dir / metadata['model'] / "BFCL_v3_simple_score.json"
                if score_file.exists():
                    with open(score_file, 'r') as f:
                        score_data = json.load(f)
                    
                    # Extract accuracy from the first entry (summary)
                    if isinstance(score_data, list) and len(score_data) > 0:
                        summary = score_data[0]
                        if 'accuracy' in summary:
                            accuracy = float(summary['accuracy'])
                            print(f"✓ Evaluation successful: {accuracy:.3f}")
                            return accuracy
                    
                    print(f"Warning: Could not find accuracy in score file")
                    return None
                else:
                    print(f"Warning: Score file not found: {score_file}")
                    return None
            else:
                print(f"✗ Evaluation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"✗ Evaluation timeout")
            return None
        except Exception as e:
            print(f"✗ Evaluation error: {e}")
            return None
    
    def evaluate_all_results(self) -> List[Dict]:
        """Evaluate all results."""
        if not self.result_dir.exists():
            print(f"Result directory {self.result_dir} does not exist!")
            return []
        
        # Find available test files
        available_tests = self.find_available_test_files()
        print(f"Found {len(available_tests)} available test file combinations:")
        for (tools, pos), file_path in sorted(available_tests.items()):
            print(f"  Tools: {tools}, Position: {pos} -> {Path(file_path).name}")
        
        results = []
        result_folders = [d for d in self.result_dir.iterdir() if d.is_dir()]
        print(f"\nFound {len(result_folders)} result folders")
        
        for folder in tqdm(result_folders, desc="Evaluating results"):
            metadata = self.parse_folder_name(folder.name)
            if metadata is None:
                continue
            
            # Check if we have a test file for this combination
            test_key = (metadata['num_tools'], metadata['position'])
            if test_key not in available_tests:
                print(f"⚠️  No test file for {metadata['model']} | Tools: {metadata['num_tools']} | Position: {metadata['position']}")
                continue
            
            test_file = available_tests[test_key]
            print(f"\n📊 Evaluating: {metadata['model']} | Tools: {metadata['num_tools']} | Position: {metadata['position']}")
            
            success_rate = self.run_bfcl_evaluation(metadata, test_file)
            
            if success_rate is not None:
                results.append({
                    'model': metadata['model'],
                    'number_of_tools': metadata['num_tools'],
                    'ground_truth_pos': metadata['position'],
                    'percentage_success': success_rate * 100  # Convert to percentage
                })
            else:
                print(f"✗ Failed to evaluate {folder.name}")
        
        return results
    
    def save_to_csv(self, results: List[Dict], output_file: str = "evaluation_results.csv"):
        """Save results to CSV."""
        if not results:
            print("No results to save!")
            return
        
        # Sort results
        results.sort(key=lambda x: (x['model'], x['number_of_tools'], x['ground_truth_pos']))
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['model', 'number_of_tools', 'ground_truth_pos', 'percentage_success']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n✅ Results saved to {output_file}")
        print(f"Total evaluations: {len(results)}")
        
        # Print summary
        print("\n=== Summary ===")
        models = set(r['model'] for r in results)
        tools = set(r['number_of_tools'] for r in results)
        positions = set(r['ground_truth_pos'] for r in results)
        
        print(f"Models: {sorted(models)}")
        print(f"Tool counts: {sorted(tools)}")
        print(f"Positions: {sorted(positions)}")
        
        if results:
            avg_success = sum(r['percentage_success'] for r in results) / len(results)
            print(f"Average success rate: {avg_success:.2f}%")
            
            # Per-model summary
            print("\n=== Per-Model Summary ===")
            for model in sorted(models):
                model_results = [r for r in results if r['model'] == model]
                model_avg = sum(r['percentage_success'] for r in model_results) / len(model_results)
                print(f"{model}: {model_avg:.2f}% (n={len(model_results)})")
            
            # Tool scaling analysis
            print("\n=== Tool Scaling Analysis ===")
            for model in sorted(models):
                print(f"\n{model}:")
                model_results = [r for r in results if r['model'] == model]
                for tools in sorted(set(r['number_of_tools'] for r in model_results)):
                    tool_results = [r for r in model_results if r['number_of_tools'] == tools]
                    if tool_results:
                        avg_for_tools = sum(r['percentage_success'] for r in tool_results) / len(tool_results)
                        positions_str = ', '.join(str(r['ground_truth_pos']) for r in sorted(tool_results, key=lambda x: x['ground_truth_pos']))
                        print(f"  {tools} tools: {avg_for_tools:.2f}% (positions: {positions_str})")


def main():
    parser = argparse.ArgumentParser(description="Final evaluation of tool scaling test results")
    parser.add_argument("--result-dir", default="result", help="Directory containing result folders")
    parser.add_argument("--score-dir", default="score_final", help="Directory to store score files")
    parser.add_argument("--output", default="evaluation_results_final.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print("=== Final Tool Scaling Test Evaluation ===")
    print(f"Result directory: {args.result_dir}")
    print(f"Score directory: {args.score_dir}")
    print(f"Output file: {args.output}")
    
    evaluator = FinalEvaluator(args.result_dir, args.score_dir)
    results = evaluator.evaluate_all_results()
    
    if results:
        evaluator.save_to_csv(results, args.output)
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()