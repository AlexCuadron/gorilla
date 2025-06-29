#!/usr/bin/env python3
"""
Simple evaluation script for tool scaling test results.

This script directly compares model outputs with ground truth without using
the complex BFCL evaluation framework.
"""

import os
import re
import json
import csv
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm


class SimpleEvaluator:
    def __init__(self, result_dir: str = "result"):
        self.result_dir = Path(result_dir)
        self.data_dir = Path("bfcl_eval/data")
        self.possible_answer_dir = self.data_dir / "possible_answer"
        
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
    
    def load_json_lines(self, file_path: Path) -> List[Dict]:
        """Load JSON lines file."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def normalize_function_call(self, func_call: str) -> str:
        """Normalize function call for comparison."""
        try:
            # Remove brackets if present
            func_call = func_call.strip()
            if func_call.startswith('[') and func_call.endswith(']'):
                func_call = func_call[1:-1]
            
            # Parse as Python AST to normalize
            parsed = ast.parse(func_call, mode='eval')
            return ast.unparse(parsed)
        except:
            # If parsing fails, return cleaned version
            return func_call.strip()
    
    def extract_function_calls(self, result_text: str) -> List[str]:
        """Extract function calls from model result."""
        try:
            # Try to parse as JSON first
            if result_text.startswith('[') and result_text.endswith(']'):
                parsed = json.loads(result_text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
                else:
                    return [str(parsed)]
            
            # Try to parse as Python list
            try:
                parsed = ast.literal_eval(result_text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
                else:
                    return [str(parsed)]
            except:
                pass
            
            # If it's a single function call
            return [result_text.strip()]
            
        except Exception as e:
            print(f"Warning: Could not parse result: {result_text[:100]}... Error: {e}")
            return [result_text.strip()]
    
    def compare_function_calls(self, model_call: str, ground_truth_calls: List[str]) -> bool:
        """Compare model function call with ground truth."""
        try:
            normalized_model = self.normalize_function_call(model_call)
            
            for gt_call in ground_truth_calls:
                try:
                    normalized_gt = self.normalize_function_call(gt_call)
                    if normalized_model == normalized_gt:
                        return True
                except:
                    # Try direct string comparison as fallback
                    if model_call.strip() == gt_call.strip():
                        return True
            
            return False
        except Exception as e:
            print(f"Warning: Error comparing calls: {e}")
            return False
    
    def evaluate_single_result(self, metadata: Dict) -> Optional[float]:
        """Evaluate a single result."""
        # Find result file
        result_dir = self.result_dir / f"{metadata['model']}_{metadata['num_tools']}_{metadata['position']}_{metadata['date']}"
        model_subdir = result_dir / metadata['model']
        result_file = model_subdir / "BFCL_v3_simple_result.json"
        
        if not result_file.exists():
            print(f"Warning: Result file not found: {result_file}")
            return None
        
        # Find test files
        test_file = self.data_dir / f"BFCL_v3_simple_tools_{metadata['num_tools']}_pos_{metadata['position']}.json"
        possible_answer_file = self.possible_answer_dir / "BFCL_v3_simple.json"
        
        if not test_file.exists():
            print(f"Warning: Test file not found: {test_file}")
            return None
        
        if not possible_answer_file.exists():
            print(f"Warning: Answer file not found: {possible_answer_file}")
            return None
        
        try:
            # Load data
            model_results = self.load_json_lines(result_file)
            test_data = self.load_json_lines(test_file)
            possible_answers = self.load_json_lines(possible_answer_file)
            
            # Create lookup for answers
            answer_lookup = {item['id']: item['ground_truth'] for item in possible_answers}
            
            correct_count = 0
            total_count = 0
            
            for model_result in model_results:
                result_id = model_result['id']
                model_output = model_result['result']
                
                if result_id in answer_lookup:
                    ground_truth = answer_lookup[result_id]
                    
                    # Extract function calls from model output
                    model_calls = self.extract_function_calls(model_output)
                    
                    # Check if any model call matches ground truth
                    is_correct = False
                    for model_call in model_calls:
                        if self.compare_function_calls(model_call, ground_truth):
                            is_correct = True
                            break
                    
                    if is_correct:
                        correct_count += 1
                    
                    total_count += 1
            
            if total_count == 0:
                print(f"Warning: No valid test cases found for {metadata['model']}_{metadata['num_tools']}_{metadata['position']}")
                return None
            
            accuracy = correct_count / total_count
            print(f"✓ {metadata['model']} | Tools: {metadata['num_tools']} | Pos: {metadata['position']} | Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
            
            return accuracy
            
        except Exception as e:
            print(f"✗ Error evaluating {metadata['model']}_{metadata['num_tools']}_{metadata['position']}: {e}")
            return None
    
    def evaluate_all_results(self) -> List[Dict]:
        """Evaluate all results."""
        if not self.result_dir.exists():
            print(f"Result directory {self.result_dir} does not exist!")
            return []
        
        results = []
        result_folders = [d for d in self.result_dir.iterdir() if d.is_dir()]
        print(f"Found {len(result_folders)} result folders")
        
        for folder in tqdm(result_folders, desc="Evaluating results"):
            metadata = self.parse_folder_name(folder.name)
            if metadata is None:
                continue
            
            success_rate = self.evaluate_single_result(metadata)
            
            if success_rate is not None:
                results.append({
                    'model': metadata['model'],
                    'number_of_tools': metadata['num_tools'],
                    'ground_truth_pos': metadata['position'],
                    'percentage_success': success_rate * 100  # Convert to percentage
                })
        
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
    parser = argparse.ArgumentParser(description="Simple evaluation of tool scaling test results")
    parser.add_argument("--result-dir", default="result", help="Directory containing result folders")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print("=== Simple Tool Scaling Test Evaluation ===")
    print(f"Result directory: {args.result_dir}")
    print(f"Output file: {args.output}")
    
    evaluator = SimpleEvaluator(args.result_dir)
    results = evaluator.evaluate_all_results()
    
    if results:
        evaluator.save_to_csv(results, args.output)
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()