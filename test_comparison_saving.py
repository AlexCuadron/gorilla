#!/usr/bin/env python3
"""
Test script to demonstrate the new comparison file saving functionality.
This shows how the evaluator now saves every comparison between AST and LLM judge.
"""

import json
import os
from pathlib import Path

def demonstrate_comparison_structure():
    """
    Show the expected directory structure and file format for comparison saving.
    """
    print("🔍 BFCL Comparison File Saving Structure")
    print("=" * 50)
    
    print("\n📁 Directory Structure:")
    print("score/")
    print("├── {model_name}/")
    print("│   ├── comparisons/")
    print("│   │   ├── {test_category}_comparison_summary.json")
    print("│   │   └── {test_category}/")
    print("│   │       ├── {test_case_id}_comparison.json")
    print("│   │       ├── {test_case_id}_comparison.json")
    print("│   │       └── ...")
    print("│   └── BFCL_v3_{test_category}_score.json")
    print("└── ...")
    
    print("\n📄 Individual Comparison File Format:")
    example_comparison = {
        "id": "tool_scaling_5_simple_111",
        "model_name": "o3-mini-2025-01-31",
        "test_category": "tool_scaling_5",
        "valid": False,
        "ast_valid": True,
        "llm_valid": False,
        "agreement": False,
        "disagreement_details": {
            "ast_says": "VALID",
            "llm_says": "INVALID",
            "ast_error_type": "none",
            "llm_error_type": "parameter_mismatch"
        },
        "ast_reasoning": "Function call syntax is correct and parameters match schema",
        "llm_reasoning": "Parameter 'temperature' should be a float but got string '0.7'",
        "error_type": "comparison:disagreement_invalid",
        "prompt": "...",
        "model_result_raw": "...",
        "model_result_decoded": "...",
        "possible_answer": "..."
    }
    
    print(json.dumps(example_comparison, indent=2))
    
    print("\n📊 Summary File Format:")
    example_summary = {
        "test_category": "tool_scaling_5",
        "model_name": "o3-mini-2025-01-31",
        "total_cases": 400,
        "ast_correct": 320,
        "llm_correct": 310,
        "agreement_count": 350,
        "disagreement_count": 50,
        "ast_accuracy": 0.80,
        "llm_accuracy": 0.775,
        "agreement_rate": 0.875,
        "disagreement_breakdown": {
            "ast_valid_llm_invalid": 30,
            "ast_invalid_llm_valid": 20
        },
        "disagreement_cases": [
            {
                "id": "tool_scaling_5_simple_111",
                "ast_says": "VALID",
                "llm_says": "INVALID",
                "ast_reasoning": "Function call syntax is correct...",
                "llm_reasoning": "Parameter 'temperature' should be..."
            }
        ]
    }
    
    print(json.dumps(example_summary, indent=2))
    
    print("\n🚀 Usage:")
    print("python -m bfcl_eval evaluate \\")
    print("  --model o3-mini-2025-01-31 \\")
    print("  --test-category tool_scaling_5 \\")
    print("  --judge o3-mini-2025-01-31")
    
    print("\n💡 Benefits:")
    print("• Individual analysis of each AST vs LLM judge comparison")
    print("• Detailed disagreement analysis with reasoning from both sides")
    print("• Statistical summaries of agreement rates")
    print("• Easy identification of systematic evaluation differences")
    print("• Debugging support for evaluation methodology")

if __name__ == "__main__":
    demonstrate_comparison_structure()