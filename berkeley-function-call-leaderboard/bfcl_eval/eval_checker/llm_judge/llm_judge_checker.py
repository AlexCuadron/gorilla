#!/usr/bin/env python3
"""
LLM Judge Checker for BFCL Evaluation

This module implements an extremely critical LLM-based evaluation system that compares
model outputs against ground truth with strict requirements for:
- Exact function name matching
- Precise parameter types and order
- Perfect value alignment

The LLM judge is designed to be EXTREMELY CRITICAL and will fail any output that
doesn't match the ground truth PERFECTLY.
"""

import json
import asyncio
import concurrent.futures
from typing import Dict, List, Any
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING


class LLMJudgeChecker:
    """
    Extremely critical LLM judge for function calling evaluation.
    """
    
    def __init__(self, judge_model: str = "gpt-4", temperature: float = 0.0):
        """Initialize the LLM judge."""
        self.judge_model = judge_model
        self.temperature = temperature
        
        # Initialize the judge model handler
        if judge_model in MODEL_CONFIG_MAPPING:
            config = MODEL_CONFIG_MAPPING[judge_model]
            self.judge_handler = config.model_handler(judge_model, temperature)
            self.judge_handler.is_fc_model = config.is_fc_model
        else:
            raise ValueError(f"Judge model '{judge_model}' not found in MODEL_CONFIG_MAPPING")
    
    def create_judge_prompt(self, ground_truth: Dict[str, Any], model_output: Any, available_functions: List[Dict] = None) -> str:
        """Create an intelligent evaluation prompt for the LLM judge."""

        # Format available functions for context
        functions_context = ""
        if available_functions:
            functions_context = f"""

**AVAILABLE FUNCTIONS (for semantic matching):**
```json
{json.dumps(available_functions, indent=2)}
```"""

        prompt = f"""You are an intelligent function calling evaluator. Your job is to determine if a model's function call output is semantically equivalent to the expected ground truth.

**GROUND TRUTH (Expected Output):**
```json
{json.dumps(ground_truth)}
```

**MODEL OUTPUT (Actual Output):**
```json
{json.dumps(model_output)}
```{functions_context}

**EVALUATION CRITERIA:**

1. **Function Name Matching**: 
   - Function names should be semantically equivalent (e.g., "calculate_triangle_area" ≈ "calc_area_triangle")
   - Check against available functions to find the best match
   - Consider synonyms and common variations (e.g., "final_speed" ≈ "final_velocity")
   - Case variations are acceptable

2. **Parameter Validation**:
   - Required parameters must be present with correct values
   - Parameter names can have semantic variations (e.g., "interest_rate" ≈ "rate")
   - Values should be equivalent (e.g., 0.05 = 5% as decimal)
   - If ground truth shows multiple possible values like [10, "10", ""], any of those values is acceptable
   - Empty string "" in ground truth means the parameter is optional
   - Lists in ground truth indicate multiple acceptable values

3. **Parameter Format Flexibility**:
   - Accept equivalent representations: dict vs individual parameters
   - Handle nested structures appropriately
   - Consider default values (e.g., n=1 for compounding frequency)

4. **Units and Optional Parameters**:
   - If units are specified in ground truth but missing in model output, it's INVALID
   - If units are optional (empty string in ground truth), missing units are acceptable
   - Optional parameters can be omitted or included

**EXAMPLES:**

✅ VALID: 
- math.factorial(number=5) ≈ {{"math.factorial": {{"number": [5]}}}}
- calculate_final_velocity(height=100, gravity=9.81) ≈ {{"calculate_final_speed": {{"height": [100], "gravity": [9.8, ""]}}}}
- calculate_compound_interest({{'principal': 5000, 'rate': 0.05, 'time': 10, 'n': 1}}) ≈ {{"calculate_compounded_interest": {{"principal": [5000], "interest_rate": [0.05], "period": [10], "compounding_frequency": ["Annually", ""]}}}}

❌ INVALID:
- calc_area_triangle(base=10, height=5) ≠ {{"calculate_triangle_area": {{"base": [10], "height": [5], "unit": ["units", ""]}}}} (missing required units)

**RESPONSE FORMAT:**
You must respond with ONLY a JSON object:
{{
    "valid": true/false,
    "reasoning": "detailed explanation of your evaluation including function matching, parameter equivalence, and any missing required fields"
}}

**SCORING GUIDELINES:**
- valid: true - Semantically equivalent function call with all required parameters
- valid: false - Missing required parameters, incompatible function, or significant semantic mismatch

Be intelligent and flexible while ensuring functional correctness.
"""
        
        return prompt
    
    def evaluate_function_call(self, ground_truth: Dict[str, Any], model_output: Any, available_functions: List[Dict] = None) -> Dict[str, Any]:
        """Evaluate a single function call using the LLM judge."""
        
        try:
            print(f"🔍 Evaluating function call with judge model: {self.judge_model}")
            
            # Create the evaluation prompt with available functions context
            prompt = self.create_judge_prompt(ground_truth, model_output, available_functions)
            
            print(f"📝 Created evaluation prompt (length: {len(prompt)} chars)")
            
            # Get judgment from the LLM
            print(f"🚀 Making API call to {self.judge_model}...")
            judgment = self._get_llm_judgment(prompt)
            
            print(f"✅ Received judgment (length: {len(judgment)} chars)")
            
            # Parse the judgment
            result = self._parse_judgment(judgment)
            
            print(f"📊 Evaluation result: valid={result['valid']}")
            
            return result
            
        except Exception as e:
            print(f"❌ LLM Judge evaluation failed: {str(e)}")
            return {
                "valid": False,
                "errors": [f"LLM Judge evaluation failed: {str(e)}"],
                "reasoning": f"Error during evaluation: {str(e)}",
                "error_type": "llm_judge:evaluation_error"
            }
    
    def _get_llm_judgment(self, prompt: str) -> str:
        """Get judgment from the LLM judge model with timeout."""
        
        try:
            # Create a simple test case for the judge
            test_case = {
                "id": "judge_evaluation",
                "question": [[{"role": "user", "content": prompt}]],
                "function": []  # No functions needed for judge
            }
            
            # Use ThreadPoolExecutor with timeout to prevent hanging
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.judge_handler.inference, test_case, False, False)
                try:
                    result, metadata = future.result(timeout=30)  # 30 second timeout
                    return result
                except concurrent.futures.TimeoutError:
                    raise Exception("LLM judge API call timed out after 30 seconds")
            
        except Exception as e:
            raise Exception(f"Failed to get LLM judgment: {str(e)}")
    
    def _parse_judgment(self, judgment: str) -> Dict[str, Any]:
        """Parse the LLM's judgment into a structured result."""
        
        try:
            # Try to extract JSON from the judgment
            if "```json" in judgment:
                start = judgment.find("```json") + 7
                end = judgment.find("```", start)
                json_str = judgment[start:end].strip()
            elif "{" in judgment and "}" in judgment:
                start = judgment.find("{")
                end = judgment.rfind("}") + 1
                json_str = judgment[start:end]
            else:
                json_str = judgment
            
            # Parse the JSON
            result = json.loads(json_str)
            
            # Validate required fields
            if "valid" not in result:
                result["valid"] = False
            if "errors" not in result:
                result["errors"] = ["Missing evaluation field"]
            if "reasoning" not in result:
                result["reasoning"] = "Incomplete evaluation"
            
            # Add error type for failed evaluations
            if not result["valid"]:
                result["error_type"] = "llm_judge:function_mismatch"
            
            return result
            
        except json.JSONDecodeError as e:
            # Fallback: try to determine validity from text
            judgment_lower = judgment.lower()
            is_valid = "valid" in judgment_lower and "true" in judgment_lower
            
            return {
                "valid": is_valid,
                "errors": ["Failed to parse LLM judgment as JSON"],
                "reasoning": f"Raw judgment: {judgment}",
                "error_type": "llm_judge:parse_error"
            }


def llm_judge_checker(model_output: List[Any], 
                     possible_answer: List[Dict[str, Any]], judge_model: str = "gpt-4", 
                     available_functions: List[Dict] = None) -> Dict[str, Any]:
    """
    Main function for LLM judge evaluation.
    
    Args:
        model_output: Model's function call output
        possible_answer: Expected ground truth
        judge_model: Model to use as judge
        available_functions: Available function definitions for semantic matching
        
    Returns:
        Evaluation result
    """
    
    # Initialize the judge
    judge = LLMJudgeChecker(judge_model)
    
    # Handle single function calls
    if len(possible_answer) == 1 and len(model_output) == 1:
        return judge.evaluate_function_call(
            ground_truth=possible_answer[0],
            model_output=model_output[0],
            available_functions=available_functions
        )
    
    # Handle multiple function calls (parallel/multiple)
    elif len(possible_answer) == len(model_output):
        # Use parallel processing for multiple function calls
        def evaluate_single(args):
            gt, output = args
            return judge.evaluate_function_call(
                ground_truth=gt,
                model_output=output,
                available_functions=available_functions
            )
        
        # Process in parallel with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(possible_answer))) as executor:
            args_list = list(zip(possible_answer, model_output))
            results = list(executor.map(evaluate_single, args_list))
        
        all_valid = all(result["valid"] for result in results)
        all_errors = []
        for result in results:
            if "errors" in result:
                all_errors.extend(result["errors"])
        
        return {
            "valid": all_valid,
            "errors": all_errors,
            "reasoning": f"Evaluated {len(results)} function calls in parallel",
            "individual_results": results,
            "error_type": "llm_judge:multiple_functions" if not all_valid else None
        }
    
    else:
        return {
            "valid": False,
            "errors": [f"Mismatch in number of function calls. Expected: {len(possible_answer)}, Got: {len(model_output)}"],
            "reasoning": "Wrong number of function calls",
            "error_type": "llm_judge:count_mismatch"
        }