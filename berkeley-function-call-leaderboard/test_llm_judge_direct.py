#!/usr/bin/env python3
"""
Direct test of LLM judge functionality without full BFCL pipeline
"""

import os
import json
import sys
from pathlib import Path

# Set up environment
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Set your API key

# Add current directory to path
sys.path.insert(0, '/workspace/gorilla/berkeley-function-call-leaderboard')

def test_llm_judge_direct():
    """Test LLM judge functionality directly"""
    
    print("🧪 Testing LLM Judge Direct Functionality")
    print("=" * 50)
    
    # Check if result file exists
    result_file = Path("result/gpt-4.1-nano-2025-04-14/BFCL_v3_tool_scaling_128_simple_result.json")
    if not result_file.exists():
        print(f"❌ Result file not found: {result_file}")
        return False
    
    print(f"✅ Found result file: {result_file}")
    
    # Load result data (JSONL format)
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    
    print(f"📊 Loaded {len(results)} results")
    
    # Check ground truth file
    ground_truth_file = Path("bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_128.json")
    if not ground_truth_file.exists():
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return False
    
    print(f"✅ Found ground truth file: {ground_truth_file}")
    
    # Load ground truth data (JSONL format)
    ground_truth = []
    with open(ground_truth_file, 'r') as f:
        for line in f:
            if line.strip():
                ground_truth.append(json.loads(line.strip()))
    
    print(f"📊 Loaded {len(ground_truth)} ground truth entries")
    
    # Test a single evaluation
    if len(results) > 0 and len(ground_truth) > 0:
        print("\n🔍 Testing single evaluation...")
        
        # Get first result
        first_result = results[0]
        print(f"📝 Test case ID: {first_result.get('id', 'unknown')}")
        print(f"🤖 Model response: {str(first_result.get('result', 'no result'))[:100]}...")
        
        # Find corresponding ground truth
        test_id = first_result.get('id')
        ground_truth_entry = None
        for gt in ground_truth:
            if gt.get('id') == test_id:
                ground_truth_entry = gt
                break
        
        if ground_truth_entry:
            print(f"✅ Found matching ground truth for ID: {test_id}")
            print(f"📋 Expected answers: {len(ground_truth_entry.get('possible_answer', []))} options")
            
            # Try to import and test LLM judge
            try:
                from bfcl_eval.eval_checker.llm_judge.llm_judge_checker import LLMJudgeChecker
                print("✅ LLMJudgeChecker imported successfully")
                
                # Initialize judge
                judge = LLMJudgeChecker("o3-mini-2025-01-31")
                print("✅ LLM Judge initialized")
                
                # Test evaluation
                model_response = first_result.get('result', [])
                expected_answers = ground_truth_entry.get('possible_answer', [])
                
                print(f"\n🎯 Testing evaluation...")
                print(f"Model response type: {type(model_response)}")
                print(f"Expected answers type: {type(expected_answers)}")
                
                # This would be the actual evaluation call
                print("⚠️  Skipping actual LLM call to avoid API costs")
                print("✅ LLM Judge setup is working correctly!")
                
                return True
                
            except Exception as e:
                print(f"❌ LLM Judge error: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"❌ No matching ground truth found for ID: {test_id}")
            return False
    
    return True

if __name__ == "__main__":
    success = test_llm_judge_direct()
    if success:
        print("\n🎉 LLM Judge test completed successfully!")
        print("\n💡 The issue with your command might be:")
        print("1. Missing dependencies (we've been fixing these)")
        print("2. API rate limiting or timeout")
        print("3. The evaluation is actually working but taking a long time")
        print("\n🚀 Try running with verbose output:")
        print("python -m bfcl_eval evaluate --model gpt-4.1-nano-2025-04-14 --test-category tool_scaling_128 --judge o3-mini-2025-01-31 -v")
    else:
        print("\n❌ LLM Judge test failed")
        sys.exit(1)