#!/usr/bin/env python3
"""
Demo script for the Tool Scaling Benchmark

This script demonstrates how to use the new tool scaling benchmark that:
1. Aggregates ALL tools from Berkeley Function Calling Leaderboard
2. Uses semantic similarity to select the top-k most relevant tools for each query
3. Provides configurable tool counts (5, 10, 20, 50, 128)

Usage:
    python demo_tool_scaling.py
"""

import json
from pathlib import Path

def load_benchmark(file_path):
    """Load a benchmark file and return the test cases."""
    test_cases = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                test_case = json.loads(line.strip())
                test_cases.append(test_case)
            except json.JSONDecodeError:
                continue
    return test_cases

def analyze_benchmark(test_cases, num_tools):
    """Analyze the benchmark and show statistics."""
    print(f"\n=== Tool Scaling Benchmark Analysis ({num_tools} tools) ===")
    print(f"Total test cases: {len(test_cases)}")
    
    # Analyze first few test cases
    print(f"\nFirst 3 test cases:")
    for i, test_case in enumerate(test_cases[:3]):
        query = test_case['question'][0][0]['content']
        functions = test_case['function']
        
        print(f"\n{i+1}. Query: {query}")
        print(f"   Number of tools provided: {len(functions)}")
        print(f"   Top 3 most relevant tools:")
        for j, func in enumerate(functions[:3]):
            print(f"     {j+1}. {func['name']}: {func.get('description', 'No description')[:80]}...")

def main():
    data_dir = Path("bfcl_eval/data")
    
    # Available tool scaling benchmarks
    tool_counts = [5, 10, 20, 50, 128]
    
    print("=== Berkeley Function Calling Leaderboard - Tool Scaling Benchmark Demo ===")
    print("\nThis benchmark demonstrates semantic tool selection using OpenAI embeddings.")
    print("For each query from BFCL_v3_simple.json, we find the most semantically similar tools")
    print("from ALL tools across the entire BFCL dataset (3259+ unique tools).")
    
    for num_tools in tool_counts:
        benchmark_file = data_dir / f"BFCL_v3_tool_scaling_{num_tools}.json"
        
        if benchmark_file.exists():
            test_cases = load_benchmark(benchmark_file)
            analyze_benchmark(test_cases, num_tools)
        else:
            print(f"\nBenchmark file not found: {benchmark_file}")
            print(f"Run: python bfcl_eval/scripts/tool_scaling_benchmark_efficient.py --num_tools {num_tools}")
    
    print("\n=== How to Use with BFCL ===")
    print("1. Generate responses:")
    print("   python -m bfcl_eval generate --model your_model --test_category tool_scaling_128")
    print("\n2. Evaluate responses:")
    print("   python -m bfcl_eval evaluate --model your_model --test_category tool_scaling_128")
    print("\n3. View scores:")
    print("   python -m bfcl_eval scores")
    
    print("\n=== Available Tool Scaling Categories ===")
    print("- tool_scaling_5: Each query gets 5 most similar tools")
    print("- tool_scaling_10: Each query gets 10 most similar tools")
    print("- tool_scaling_20: Each query gets 20 most similar tools")
    print("- tool_scaling_50: Each query gets 50 most similar tools")
    print("- tool_scaling_128: Each query gets 128 most similar tools (default)")
    print("- tool_scaling: All tool scaling benchmarks")

if __name__ == "__main__":
    main()