#!/usr/bin/env python3
"""
Test script to understand the evaluation process for a single result.
"""

import subprocess
import json
from pathlib import Path

def test_single_evaluation():
    # Pick one result directory to test
    result_dir = "result/gpt-4.1-nano-2025-04-14_10_0_2025-06-27"
    score_dir = "score_test"
    
    # Create score directory
    Path(score_dir).mkdir(exist_ok=True)
    
    print(f"Testing evaluation for: {result_dir}")
    
    # Check what's in the result file
    result_file = Path(result_dir) / "gpt-4.1-nano-2025-04-14" / "BFCL_v3_simple_result.json"
    print(f"Result file: {result_file}")
    
    if result_file.exists():
        with open(result_file, 'r') as f:
            first_line = f.readline()
            result_data = json.loads(first_line)
            print(f"First result ID: {result_data['id']}")
            print(f"First result: {result_data['result']}")
    
    # Try different test categories and see what works
    test_attempts = [
        ("simple", None),
        ("custom", "bfcl_eval/data/BFCL_v3_custom_beginning.json"),
        ("simple_tools_10_pos_1", None),
    ]
    
    for test_category, custom_path in test_attempts:
        print(f"\n=== Trying test_category: {test_category} ===")
        
        cmd = [
            "python", "-m", "bfcl_eval", "evaluate",
            "--model", "gpt-4.1-nano-2025-04-14",
            "--test-category", test_category,
            "--result-dir", result_dir,
            "--score-dir", score_dir
        ]
        
        if custom_path:
            cmd.extend(["--custom-path", custom_path])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(f"Return code: {result.returncode}")
            if result.returncode == 0:
                print("✓ SUCCESS!")
                print(f"Stdout: {result.stdout}")
                
                # Check what score files were created
                score_files = list(Path(score_dir).rglob("*.json"))
                print(f"Score files created: {score_files}")
                
                for score_file in score_files:
                    with open(score_file, 'r') as f:
                        score_data = json.load(f)
                        print(f"Score data keys: {list(score_data.keys())}")
                        if isinstance(score_data, dict):
                            for key, value in score_data.items():
                                if isinstance(value, (int, float)) and 0 <= value <= 1:
                                    print(f"  {key}: {value}")
                
                break
            else:
                print(f"✗ FAILED")
                print(f"Stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("✗ TIMEOUT")
        except Exception as e:
            print(f"✗ ERROR: {e}")

if __name__ == "__main__":
    test_single_evaluation()