#!/usr/bin/env python3
"""
Proper evaluation script for tool scaling test results.

This script:
1. Parses result folder names: model_number_of_tools_position_of_ground_truth_date
2. Evaluates each result against the correct test file
3. Calculates actual success percentages
4. Builds a CSV with: model, number_of_tools, ground_truth_pos, percentage_success
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm

# Import BFCL evaluation modules
import sys
sys.path.append('.')
from bfcl_eval.eval_checker.eval_runner import get_handler, ast_file_runner
from bfcl_eval.utils import load_file


class ToolScalingEvaluator:
    def __init__(self, result_dir: str = "result"):
        self.result_dir = Path(result_dir)
        self.data_dir = Path("bfcl_eval/data")
        self.possible_answer_dir = self.data_dir / "possible_answer"
        
    def parse_folder_name(self, folder_name: str) -> Optional[Dict[str, any]]:
        """
        Parse folder name to extract: model_number_of_tools_position_of_ground_truth_date
        
        Examples:
        - gpt-4.1-nano-2025-04-14_10_0_2025-06-27
        - o3-mini-2025-01-31_50_50_2025-06-27
        
        Returns:
            Dict with keys: model, num_tools, position, date
        """
        # Pattern: model_name_tools_position_date
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
        else:
            print(f"Warning: Could not parse folder name: {folder_name}")
            return None
    
    def find_test_files(self, num_tools: int, position: int) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find the correct test file and possible answer file for given parameters.
        
        Returns:
            Tuple of (test_file_path, possible_answer_path)
        """
        # Look for test file with pattern: BFCL_v3_simple_tools_{num_tools}_pos_{position}.json
        test_file = self.data_dir / f"BFCL_v3_simple_tools_{num_tools}_pos_{position}.json"
        
        if not test_file.exists():
            print(f"Warning: Test file not found: {test_file}")
            return None, None
        
        # Look for possible answer file
        possible_answer_file = self.possible_answer_dir / "BFCL_v3_simple.json"
        
        if not possible_answer_file.exists():
            print(f"Warning: Possible answer file not found: {possible_answer_file}")
            return test_file, None
        
        return test_file, possible_answer_file
    
    def evaluate_single_result(self, metadata: Dict) -> Optional[float]:
        """
        Evaluate a single result and return success percentage.
        
        Returns:
            Success percentage (0.0 to 1.0) or None if evaluation failed
        """
        # Find result file
        result_dir = self.result_dir / f"{metadata['model']}_{metadata['num_tools']}_{metadata['position']}_{metadata['date']}"
        model_subdir = result_dir / metadata['model']
        result_file = model_subdir / "BFCL_v3_simple_result.json"
        
        if not result_file.exists():
            print(f"Warning: Result file not found: {result_file}")
            return None
        
        # Find test files
        test_file, possible_answer_file = self.find_test_files(metadata['num_tools'], metadata['position'])
        if test_file is None or possible_answer_file is None:
            return None
        
        try:
            # Load data
            model_result = load_file(result_file, sort_by_id=True)
            prompt = load_file(test_file, sort_by_id=True)
            possible_answer = load_file(possible_answer_file, sort_by_id=True)
            
            # Get model handler
            handler = get_handler(metadata['model'])
            
            # Create temporary score directory
            temp_score_dir = Path("temp_score")
            temp_score_dir.mkdir(exist_ok=True)
            
            # Run evaluation
            accuracy, total_count = ast_file_runner(
                handler=handler,
                model_result=model_result,
                prompt=prompt,
                possible_answer=possible_answer,
                language="Python",
                test_category="simple",
                model_name=metadata['model'],
                score_dir=temp_score_dir
            )
            
            print(f"✓ Evaluated {metadata['model']} with {metadata['num_tools']} tools, pos {metadata['position']}: {accuracy:.3f} ({int(accuracy*total_count)}/{total_count})")
            
            return accuracy
            
        except Exception as e:
            print(f"✗ Error evaluating {metadata['model']}_{metadata['num_tools']}_{metadata['position']}: {e}")
            return None
    
    def evaluate_all_results(self) -> List[Dict]:
        """
        Evaluate all results in the result directory.
        
        Returns:
            List of evaluation results
        """
        if not self.result_dir.exists():
            print(f"Result directory {self.result_dir} does not exist!")
            return []
        
        results = []
        
        # Get all result folders
        result_folders = [d for d in self.result_dir.iterdir() if d.is_dir()]
        print(f"Found {len(result_folders)} result folders")
        
        for folder in tqdm(result_folders, desc="Evaluating results"):
            metadata = self.parse_folder_name(folder.name)
            if metadata is None:
                continue
            
            print(f"\nEvaluating: {metadata['model']} | Tools: {metadata['num_tools']} | Position: {metadata['position']}")
            
            success_rate = self.evaluate_single_result(metadata)
            
            if success_rate is not None:
                results.append({
                    'model': metadata['model'],
                    'number_of_tools': metadata['num_tools'],
                    'ground_truth_pos': metadata['position'],
                    'percentage_success': success_rate  # Keep as 0-1 range for now
                })
            else:
                print(f"✗ Failed to evaluate {folder.name}")
        
        return results
    
    def save_to_csv(self, results: List[Dict], output_file: str = "evaluation_results.csv"):
        """
        Save results to CSV file.
        """
        if not results:
            print("No results to save!")
            return
        
        # Convert success rate to percentage (0-100 range)
        for result in results:
            result['percentage_success'] = result['percentage_success'] * 100
        
        # Sort results
        results.sort(key=lambda x: (x['model'], x['number_of_tools'], x['ground_truth_pos']))
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['model', 'number_of_tools', 'ground_truth_pos', 'percentage_success']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n✓ Results saved to {output_file}")
        print(f"Total evaluations: {len(results)}")
        
        # Print summary
        print("\n=== Summary ===")
        models = set(r['model'] for r in results)
        tools = set(r['number_of_tools'] for r in results)
        positions = set(r['ground_truth_pos'] for r in results)
        
        print(f"Models: {sorted(models)}")
        print(f"Tool counts: {sorted(tools)}")
        print(f"Positions: {sorted(positions)}")
        
        avg_success = sum(r['percentage_success'] for r in results) / len(results)
        print(f"Average success rate: {avg_success:.2f}%")
        
        # Per-model summary
        print("\n=== Per-Model Summary ===")
        for model in sorted(models):
            model_results = [r for r in results if r['model'] == model]
            model_avg = sum(r['percentage_success'] for r in model_results) / len(model_results)
            print(f"{model}: {model_avg:.2f}% (n={len(model_results)})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool scaling test results")
    parser.add_argument("--result-dir", default="result", help="Directory containing result folders")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print("=== Tool Scaling Test Evaluation ===")
    print(f"Result directory: {args.result_dir}")
    print(f"Output file: {args.output}")
    
    evaluator = ToolScalingEvaluator(args.result_dir)
    results = evaluator.evaluate_all_results()
    
    if results:
        evaluator.save_to_csv(results, args.output)
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()