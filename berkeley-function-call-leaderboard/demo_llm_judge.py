#!/usr/bin/env python3
"""
Demo script showing LLM Judge integration with BFCL evaluation
"""

import json

def demo_llm_judge_integration():
    """Demonstrate how the LLM judge integrates with BFCL evaluation."""
    
    print("🎯 LLM Judge Integration Demo")
    print("=" * 50)
    
    print("\n📋 Overview:")
    print("The LLM judge has been integrated into BFCL evaluation with the following features:")
    
    print("\n🔧 Usage:")
    print("python -m bfcl_eval evaluate --test-category tool_scaling --judge gpt-4")
    print("python -m bfcl_eval evaluate --test-category simple --judge gpt-3.5-turbo")
    
    print("\n⚙️  Key Features:")
    print("1. EXTREMELY CRITICAL evaluation of function calls")
    print("2. Exact function name matching (case-sensitive)")
    print("3. Precise parameter validation")
    print("4. Perfect value alignment with ground truth")
    print("5. Support for all BFCL test categories")
    
    print("\n📊 Ground Truth Integration:")
    print("✅ Tool scaling ground truth files generated:")
    print("   - BFCL_v3_tool_scaling_5.json (400 test cases)")
    print("   - BFCL_v3_tool_scaling_10.json (400 test cases)")
    print("   - BFCL_v3_tool_scaling_20.json (400 test cases)")
    print("   - BFCL_v3_tool_scaling_50.json (400 test cases)")
    print("   - BFCL_v3_tool_scaling_128.json (400 test cases)")
    
    print("\n🎯 Evaluation Criteria (EXTREMELY STRICT):")
    print("1. Function Name: Must match EXACTLY (no variations)")
    print("2. Parameter Names: Must match EXACTLY (case-sensitive)")
    print("3. Parameter Values: Must be in allowed ground truth list")
    print("4. No Extra Parameters: Only ground truth parameters allowed")
    print("5. Type Validation: Must match expected types")
    
    print("\n📝 Example Ground Truth:")
    example_gt = {
        "id": "tool_scaling_5_simple_0",
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    }
    print(json.dumps(example_gt, indent=2))
    
    print("\n✅ Model Output Examples:")
    print("VALID:")
    valid_output = {"calculate_triangle_area": {"base": 10, "height": 5, "unit": "units"}}
    print(f"  {json.dumps(valid_output)}")
    
    print("\nINVALID:")
    print("  Wrong function: {'calculate_area': {'base': 10, 'height': 5}}")
    print("  Wrong parameter: {'calculate_triangle_area': {'base': 15, 'height': 5}}")
    print("  Extra parameter: {'calculate_triangle_area': {'base': 10, 'height': 5, 'extra': 'value'}}")
    
    print("\n🚀 Integration Status:")
    print("✅ LLM judge checker implemented")
    print("✅ Ground truth files generated")
    print("✅ Category mappings updated")
    print("✅ CLI flag --judge added")
    print("✅ Evaluation pipeline integrated")
    
    print("\n🎉 Ready for use!")
    print("The tool scaling benchmark can now be evaluated with extreme precision using LLM judges.")

if __name__ == "__main__":
    demo_llm_judge_integration()