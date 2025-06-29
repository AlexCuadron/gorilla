# LLM Judge Integration for BFCL Tool Scaling Benchmark

## Overview

This document describes the integration of an extremely critical LLM judge for evaluating function calling performance in the Berkeley Function Call Leaderboard (BFCL), specifically designed for the tool scaling benchmark.

## Key Features

### 🎯 Extremely Critical Evaluation
The LLM judge is designed to be **EXTREMELY CRITICAL** and will fail any output that doesn't match the ground truth **PERFECTLY**:

- **Function Name**: Must match EXACTLY (case-sensitive, no variations allowed)
- **Parameter Names**: Must match EXACTLY (case-sensitive, no variations allowed)  
- **Parameter Types**: Must match the expected types EXACTLY
- **Parameter Values**: Must match the expected values EXACTLY or be in the allowed list
- **No Extra Parameters**: Only parameters that are in the ground truth are allowed

### 🔧 Usage

```bash
# Evaluate tool scaling benchmarks with LLM judge
python -m bfcl_eval evaluate --test-category tool_scaling --judge gpt-4

# Evaluate specific tool scaling benchmark
python -m bfcl_eval evaluate --test-category tool_scaling_128 --judge gpt-4

# Evaluate simple benchmark with LLM judge
python -m bfcl_eval evaluate --test-category simple --judge gpt-3.5-turbo

# Evaluate all benchmarks with LLM judge
python -m bfcl_eval evaluate --test-category all --judge gpt-4
```

### 📊 Ground Truth Integration

Ground truth files have been generated for all tool scaling benchmarks:

- `BFCL_v3_tool_scaling_5.json` (400 test cases, 5 tools each)
- `BFCL_v3_tool_scaling_10.json` (400 test cases, 10 tools each)
- `BFCL_v3_tool_scaling_20.json` (400 test cases, 20 tools each)
- `BFCL_v3_tool_scaling_50.json` (400 test cases, 50 tools each)
- `BFCL_v3_tool_scaling_128.json` (400 test cases, 128 tools each)

Each ground truth entry follows the BFCL format:
```json
{
  "id": "tool_scaling_128_simple_0",
  "ground_truth": [
    {
      "calculate_triangle_area": {
        "base": [10],
        "height": [5], 
        "unit": ["units", ""]
      }
    }
  ]
}
```

## Implementation Details

### 🏗️ Architecture

1. **LLM Judge Checker** (`bfcl_eval/eval_checker/llm_judge/llm_judge_checker.py`)
   - Implements extremely critical evaluation logic
   - Uses configurable LLM models (GPT-4, GPT-3.5-turbo, etc.)
   - Provides detailed error reporting and reasoning

2. **Integration with BFCL Pipeline**
   - Modified `eval_runner.py` to support `--judge` flag
   - Updated CLI interface in `__main__.py`
   - Integrated with existing evaluation workflow

3. **Category Mapping Updates**
   - Added tool scaling categories to all relevant collections
   - Supports evaluation alongside existing BFCL benchmarks

### 🎯 Evaluation Criteria

The LLM judge uses the following extremely strict criteria:

#### 1. Function Name Matching
```python
# VALID
{"calculate_triangle_area": {"base": 10, "height": 5}}

# INVALID - wrong function name
{"calculate_area": {"base": 10, "height": 5}}
{"triangleArea": {"base": 10, "height": 5}}
```

#### 2. Parameter Validation
```python
# Ground truth allows multiple values
"base": [10, "10"]  # Both 10 and "10" are acceptable

# VALID
{"calculate_triangle_area": {"base": 10, "height": 5}}
{"calculate_triangle_area": {"base": "10", "height": 5}}

# INVALID - wrong value
{"calculate_triangle_area": {"base": 15, "height": 5}}
```

#### 3. No Extra Parameters
```python
# VALID - only required parameters
{"calculate_triangle_area": {"base": 10, "height": 5}}

# INVALID - extra parameter
{"calculate_triangle_area": {"base": 10, "height": 5, "extra": "value"}}
```

### 🚀 Performance Characteristics

- **Precision**: Extremely high precision due to LLM-based evaluation
- **Flexibility**: Can handle semantic equivalence while maintaining strictness
- **Scalability**: Works with all tool scaling benchmarks (5 to 128 tools)
- **Consistency**: Temperature set to 0.0 for consistent evaluation

## Files Modified/Added

### New Files
- `bfcl_eval/eval_checker/llm_judge/__init__.py`
- `bfcl_eval/eval_checker/llm_judge/llm_judge_checker.py`
- `bfcl_eval/data/possible_answer/BFCL_v3_tool_scaling_*.json` (5 files)
- `generate_tool_scaling_ground_truth.py`
- `demo_llm_judge.py`
- `test_llm_judge.py`

### Modified Files
- `bfcl_eval/__main__.py` - Added `--judge` CLI flag
- `bfcl_eval/eval_checker/eval_runner.py` - Integrated LLM judge evaluation
- `bfcl_eval/constants/category_mapping.py` - Added tool scaling categories

## Example Usage Scenarios

### 1. Research Evaluation
```bash
# Compare AST checker vs LLM judge on simple benchmark
python -m bfcl_eval evaluate --test-category simple --model gpt-4
python -m bfcl_eval evaluate --test-category simple --model gpt-4 --judge gpt-4
```

### 2. Tool Scaling Analysis
```bash
# Evaluate how model performance degrades with tool count
python -m bfcl_eval evaluate --test-category tool_scaling_5 --judge gpt-4
python -m bfcl_eval evaluate --test-category tool_scaling_10 --judge gpt-4
python -m bfcl_eval evaluate --test-category tool_scaling_20 --judge gpt-4
python -m bfcl_eval evaluate --test-category tool_scaling_50 --judge gpt-4
python -m bfcl_eval evaluate --test-category tool_scaling_128 --judge gpt-4
```

### 3. Model Comparison
```bash
# Compare different models on tool scaling with LLM judge
python -m bfcl_eval evaluate --test-category tool_scaling --model gpt-4 --judge gpt-4
python -m bfcl_eval evaluate --test-category tool_scaling --model claude-3 --judge gpt-4
```

## Benefits

1. **Extreme Precision**: LLM judge catches subtle errors that AST checker might miss
2. **Semantic Understanding**: Can evaluate semantic equivalence while maintaining strictness
3. **Detailed Feedback**: Provides comprehensive error analysis and reasoning
4. **Flexibility**: Works with any LLM model supported by BFCL
5. **Integration**: Seamlessly integrates with existing BFCL workflow

## Future Enhancements

1. **Multi-Judge Consensus**: Use multiple LLM judges for even higher reliability
2. **Custom Prompts**: Allow custom evaluation prompts for specific use cases
3. **Confidence Scoring**: Add confidence scores to judge evaluations
4. **Batch Processing**: Optimize for large-scale evaluations

## Conclusion

The LLM judge integration provides an extremely critical and precise evaluation mechanism for the BFCL tool scaling benchmark. It maintains the highest standards of accuracy while providing detailed feedback for model improvement and research analysis.