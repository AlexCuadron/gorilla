#!/usr/bin/env python3
"""
Generate ground truth files for tool scaling benchmarks.

This script creates the possible_answer files needed for BFCL evaluation
by extracting the ground truth from the original BFCL_v3_simple.json possible answers
and mapping them to the tool scaling benchmark IDs.
"""

import json
import os
from pathlib import Path

def load_simple_ground_truth():
    """Load the ground truth from BFCL_v3_simple.json possible answer file."""
    simple_answer_file = Path("bfcl_eval/data/possible_answer/BFCL_v3_simple.json")
    
    ground_truth_map = {}
    with open(simple_answer_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Extract the simple_X number from the ID
            simple_id = data["id"]
            ground_truth_map[simple_id] = data["ground_truth"]
    
    return ground_truth_map

def generate_tool_scaling_ground_truth(tool_counts=[5, 10, 20, 50, 128]):
    """Generate ground truth files for all tool scaling benchmarks."""
    
    print("🔍 Loading ground truth from BFCL_v3_simple.json...")
    simple_ground_truth = load_simple_ground_truth()
    
    for tool_count in tool_counts:
        print(f"📝 Generating ground truth for tool_scaling_{tool_count}...")
        
        # Load the tool scaling benchmark file
        benchmark_file = f"BFCL_v3_tool_scaling_{tool_count}.json"
        output_file = f"bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_{tool_count}.json"
        
        if not os.path.exists(benchmark_file):
            print(f"❌ Benchmark file {benchmark_file} not found. Skipping...")
            continue
        
        ground_truth_entries = []
        
        with open(benchmark_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                test_id = data["id"]
                
                # Extract the original simple test ID from the tool scaling ID
                # Format: tool_scaling_128_simple_0 -> simple_0
                parts = test_id.split('_')
                if len(parts) >= 4 and parts[-2] == "simple":
                    simple_id = f"simple_{parts[-1]}"
                    
                    if simple_id in simple_ground_truth:
                        ground_truth_entry = {
                            "id": test_id,
                            "ground_truth": simple_ground_truth[simple_id]
                        }
                        ground_truth_entries.append(ground_truth_entry)
                    else:
                        print(f"⚠️  Warning: No ground truth found for {simple_id}")
                else:
                    print(f"⚠️  Warning: Unexpected ID format: {test_id}")
        
        # Write the ground truth file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for entry in ground_truth_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✅ Generated {len(ground_truth_entries)} ground truth entries for tool_scaling_{tool_count}")
        print(f"📁 Saved to: {output_file}")

if __name__ == "__main__":
    print("🚀 Generating ground truth files for tool scaling benchmarks...")
    generate_tool_scaling_ground_truth()
    print("🎉 Ground truth generation completed!")