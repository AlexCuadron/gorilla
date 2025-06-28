#!/usr/bin/env python3
"""
Test the parallel LLM judge functionality
"""

import os
import sys
import json
import time

# Set up environment
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Set your API key

# Add current directory to path
sys.path.insert(0, '/workspace/gorilla/berkeley-function-call-leaderboard')

def test_parallel_judge():
    """Test the parallel LLM judge functionality"""
    
    print("🧪 Testing Parallel LLM Judge")
    print("=" * 40)
    
    try:
        # Load a few test cases from the result file
        result_file = "result/gpt-4.1-nano-2025-04-14/BFCL_v3_tool_scaling_128_simple_result.json"
        ground_truth_file = "bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_128.json"
        
        # Load first 3 results for testing
        results = []
        with open(result_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Only test first 3
                    break
                if line.strip():
                    results.append(json.loads(line.strip()))
        
        # Load corresponding ground truth
        ground_truth = []
        with open(ground_truth_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Only test first 3
                    break
                if line.strip():
                    ground_truth.append(json.loads(line.strip()))
        
        print(f"📊 Loaded {len(results)} test cases")
        
        # Import the LLM judge
        from bfcl_eval.eval_checker.llm_judge.llm_judge_checker import llm_judge_checker
        
        # Test with first case
        if len(results) > 0 and len(ground_truth) > 0:
            print(f"\n🔍 Testing single evaluation...")
            
            test_result = results[0]
            test_gt = ground_truth[0]
            
            print(f"Test ID: {test_result.get('id')}")
            print(f"Model output: {test_result.get('result', [])}")
            print(f"Expected: {test_gt.get('possible_answer', [])}")
            
            # Mock function description (simplified)
            func_description = [{"name": "test_function", "parameters": []}]
            
            start_time = time.time()
            
            # Run evaluation with timeout protection
            try:
                evaluation_result = llm_judge_checker(
                    func_description=func_description,
                    model_output=test_result.get('result', []),
                    possible_answer=test_gt.get('possible_answer', []),
                    judge_model="o3-mini-2025-01-31"
                )
                
                end_time = time.time()
                
                print(f"✅ Evaluation completed in {end_time - start_time:.2f} seconds")
                print(f"📊 Result: {evaluation_result}")
                
                return True
                
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        else:
            print("❌ No test data available")
            return False
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parallel_judge()
    if success:
        print("\n🎉 Parallel LLM Judge test successful!")
        print("\n🚀 Now you can run the full evaluation:")
        print("python -m bfcl_eval evaluate --model gpt-4.1-nano-2025-04-14 --test-category tool_scaling_128 --judge o3-mini-2025-01-31")
    else:
        print("\n❌ Parallel LLM Judge test failed!")