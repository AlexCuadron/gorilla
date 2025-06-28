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
    
    def create_judge_prompt(self, ground_truth: Dict[str, Any], model_output: Any, 
                          available_functions: List[Dict[str, Any]]) -> str:
        """Create an extremely critical evaluation prompt for the LLM judge."""
        
        prompt = f"""You are an EXTREMELY CRITICAL function calling evaluator. Your job is to determine if a model's function call output EXACTLY matches the expected ground truth.

You must be EXTREMELY STRICT about:
1. **Function Name**: Must match EXACTLY (case-sensitive, no variations allowed)
2. **Parameter Names**: Must match EXACTLY (case-sensitive, no variations allowed)  
3. **Parameter Types**: Must match the expected types EXACTLY
4. **Parameter Values**: Must match the expected values EXACTLY or be in the allowed list

**GROUND TRUTH (Expected Output):**
```json
{json.dumps(ground_truth, indent=2)}
```

**MODEL OUTPUT (Actual Output):**
```json
{json.dumps(model_output, indent=2)}
```

**AVAILABLE FUNCTIONS:**
```json
{json.dumps(available_functions, indent=2)}
```

**EVALUATION CRITERIA:**

1. **Function Name Matching**: 
   - The function name must be EXACTLY the same as in ground truth
   - No synonyms, abbreviations, or variations are acceptable
   - Case sensitivity matters

2. **Parameter Validation**:
   - Every required parameter must be present
   - Parameter names must match EXACTLY
   - Parameter values must be in the allowed list from ground truth
   - If ground truth shows multiple possible values like [10, "10"], any of those values is acceptable

3. **No Extra Parameters**:
   - The model should not include parameters not in ground truth
   - Only parameters that are in the ground truth are allowed

**RESPONSE FORMAT:**
You must respond with ONLY a JSON object:
{{
    "valid": true/false,
    "errors": ["list of specific errors found"],
    "reasoning": "detailed explanation of your evaluation"
}}

**SCORING GUIDELINES:**
- valid: true - Perfect match in function name and all parameters
- valid: false - Any mismatch in function name or parameters

Be EXTREMELY CRITICAL. If there's ANY doubt, mark it as invalid.
"""
        
        return prompt
    
    def evaluate_function_call(self, ground_truth: Dict[str, Any], model_output: Any,
                             available_functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single function call using the LLM judge."""
        
        try:
            # Create the evaluation prompt
            prompt = self.create_judge_prompt(ground_truth, model_output, available_functions)
            
            # Get judgment from the LLM
            judgment = self._get_llm_judgment(prompt)
            
            # Parse the judgment
            result = self._parse_judgment(judgment)
            
            return result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"LLM Judge evaluation failed: {str(e)}"],
                "reasoning": f"Error during evaluation: {str(e)}",
                "error_type": "llm_judge:evaluation_error"
            }
    
    def _get_llm_judgment(self, prompt: str) -> str:
        """Get judgment from the LLM judge model."""
        
        try:
            # Create a simple test case for the judge
            test_case = {
                "id": "judge_evaluation",
                "question": [[{"role": "user", "content": prompt}]],
                "function": []  # No functions needed for judge
            }
            
            # Get response from judge model
            result, metadata = self.judge_handler.inference(test_case, False, False)
            
            return result
            
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


def llm_judge_checker(func_description: List[Dict[str, Any]], model_output: List[Any], 
                     possible_answer: List[Dict[str, Any]], judge_model: str = "gpt-4") -> Dict[str, Any]:
    """
    Main function for LLM judge evaluation.
    
    Args:
        func_description: Available function definitions
        model_output: Model's function call output
        possible_answer: Expected ground truth
        judge_model: Model to use as judge
        
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
            available_functions=func_description
        )
    
    # Handle multiple function calls (parallel/multiple)
    elif len(possible_answer) == len(model_output):
        results = []
        all_valid = True
        all_errors = []
        
        for i, (gt, output) in enumerate(zip(possible_answer, model_output)):
            result = judge.evaluate_function_call(
                ground_truth=gt,
                model_output=output,
                available_functions=func_description
            )
            
            results.append(result)
            all_valid = all_valid and result["valid"]
            all_errors.extend(result["errors"])
        
        return {
            "valid": all_valid,
            "errors": all_errors,
            "reasoning": f"Evaluated {len(results)} function calls",
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