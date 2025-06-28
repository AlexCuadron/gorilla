#!/usr/bin/env python3
"""
Debug the evaluation process to see why it's not finding results
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/workspace/gorilla/berkeley-function-call-leaderboard')

def debug_evaluation():
    """Debug the evaluation process"""
    
    print("🔍 Debugging Evaluation Process")
    print("=" * 50)
    
    # Check the result directory structure
    from bfcl_eval.constants.eval_config import RESULT_PATH
    
    print(f"📁 Result path: {RESULT_PATH}")
    print(f"📁 Result path exists: {RESULT_PATH.exists()}")
    
    if RESULT_PATH.exists():
        print(f"📁 Contents of result directory:")
        for item in RESULT_PATH.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            if item.is_dir():
                print(f"    Contents:")
                for subitem in item.iterdir():
                    print(f"      - {subitem.name}")
    
    # Check model config
    print(f"\n🤖 Checking model config...")
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        model_name = "gpt-4.1-nano-2025-04-14"
        if model_name in MODEL_CONFIG_MAPPING:
            print(f"✅ Model '{model_name}' found in config")
            config = MODEL_CONFIG_MAPPING[model_name]
            print(f"   - Display name: {config.display_name}")
            print(f"   - Is FC model: {config.is_fc_model}")
        else:
            print(f"❌ Model '{model_name}' NOT found in config")
            print(f"Available models with 'gpt' in name:")
            for k in sorted(MODEL_CONFIG_MAPPING.keys()):
                if 'gpt' in k.lower():
                    print(f"  - {k}")
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
    
    # Check test categories
    print(f"\n📋 Checking test categories...")
    try:
        from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
        test_category = "tool_scaling_128"
        if test_category in TEST_COLLECTION_MAPPING:
            print(f"✅ Test category '{test_category}' found")
            print(f"   - Maps to: {TEST_COLLECTION_MAPPING[test_category]}")
        else:
            print(f"❌ Test category '{test_category}' NOT found")
            print(f"Available test categories:")
            for k in sorted(TEST_COLLECTION_MAPPING.keys()):
                print(f"  - {k}")
    except Exception as e:
        print(f"❌ Error loading test categories: {e}")
    
    # Check if the result file matches expected pattern
    print(f"\n📄 Checking result file pattern...")
    result_dir = RESULT_PATH / "gpt-4.1-nano-2025-04-14"
    if result_dir.exists():
        print(f"✅ Model result directory exists: {result_dir}")
        for json_file in result_dir.glob("*.json"):
            print(f"  📄 Found result file: {json_file.name}")
            
            # Extract test category from filename
            try:
                from bfcl_eval.eval_checker.eval_runner_helper import extract_test_category
                test_cat = extract_test_category(json_file)
                print(f"     - Extracted test category: {test_cat}")
            except Exception as e:
                print(f"     - Error extracting test category: {e}")
    else:
        print(f"❌ Model result directory does not exist: {result_dir}")
    
    # Try to run a minimal evaluation
    print(f"\n🧪 Testing minimal evaluation...")
    try:
        from bfcl_eval.eval_checker.eval_runner import main
        
        # Set environment variable for API key
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # You'll need to set this
        
        print("Calling evaluation main function...")
        main(
            model=["gpt-4.1-nano-2025-04-14"],
            test_categories=["tool_scaling_128"],
            result_dir=None,
            score_dir=None,
            judge_model="o3-mini-2025-01-31"
        )
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_evaluation()