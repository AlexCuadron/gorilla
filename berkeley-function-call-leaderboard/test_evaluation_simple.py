#!/usr/bin/env python3
"""
Simple test script to verify the evaluation works without all dependencies
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/workspace/gorilla/berkeley-function-call-leaderboard')

def test_evaluation_command():
    """Test the evaluation command with proper model name."""
    
    print("🧪 Testing BFCL evaluation with LLM judge...")
    print("=" * 50)
    
    # Check if ground truth files exist
    ground_truth_files = [
        "bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_5.json",
        "bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_10.json", 
        "bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_20.json",
        "bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_50.json",
        "bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_128.json"
    ]
    
    print("\n📁 Checking ground truth files:")
    for file_path in ground_truth_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Check if prompt files exist
    prompt_files = [
        "bfcl_eval/data/prompts/BFCL_v3_tool_scaling_5.json",
        "bfcl_eval/data/prompts/BFCL_v3_tool_scaling_10.json",
        "bfcl_eval/data/prompts/BFCL_v3_tool_scaling_20.json", 
        "bfcl_eval/data/prompts/BFCL_v3_tool_scaling_50.json",
        "bfcl_eval/data/prompts/BFCL_v3_tool_scaling_128.json"
    ]
    
    print("\n📝 Checking prompt files:")
    for file_path in prompt_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
    
    print("\n🔧 Correct command usage:")
    print("python -m bfcl_eval evaluate --test-category tool_scaling_128 --judge o3-mini-2025-01-31")
    print("python -m bfcl_eval evaluate --test-category tool_scaling_128 --judge gpt-4")
    print("python -m bfcl_eval evaluate --test-category tool_scaling_128 --judge gpt-3.5-turbo")
    
    print("\n❌ Incorrect model name used:")
    print("o3-2025-04-16 (this model doesn't exist in BFCL config)")
    
    print("\n✅ Available o3 models:")
    print("o3-mini-2025-01-31")
    print("o3-mini-2025-01-31-FC")
    
    print("\n🚀 To fix the issue:")
    print("1. Use correct model name: o3-mini-2025-01-31")
    print("2. Make sure you have OpenAI API key set")
    print("3. Install missing dependencies if needed")
    
    return True

if __name__ == "__main__":
    test_evaluation_command()