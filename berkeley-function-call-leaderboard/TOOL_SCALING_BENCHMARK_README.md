# Tool Scaling Benchmark for BFCL

## 🎯 Overview

The Tool Scaling Benchmark is a comprehensive evaluation framework that tests how well language models perform function calling when presented with varying numbers of available tools. This benchmark uses semantic similarity to select the most relevant tools for each query, creating realistic scenarios where models must choose from large tool sets.

## 📊 Key Features

### **🔧 Comprehensive Tool Aggregation**
- **33,090 unique tools** aggregated from ALL Gorilla datasets:
  - BFCL files: 3,365 tools
  - Multi-turn func doc: 129 tools  
  - OpenFunctions v1: 3,593 tools
  - APIZoo: 7,276 tools
  - APIBench: 17,003 tools
  - API directory: 1,724 tools

### **🚀 High-Performance Implementation**
- **Parallel processing** with 8 concurrent workers
- **Optimized batch processing** (200 items per batch)
- **Pre-computed embeddings matrix** for vectorized similarity computation
- **Automatic caching** to avoid recomputing embeddings
- **70 seconds** to compute embeddings for all 33,090 tools

### **🎯 Semantic Tool Selection**
- Uses OpenAI's `text-embedding-3-large` for semantic similarity
- **Cosine similarity** ranking to find most relevant tools
- **Configurable tool count** (default: 128 tools per query)
- Ensures tools are semantically relevant to each query

### **✅ Robust Duplicate Detection**
- JSON-based hashing with `sort_keys=True` for exact matching
- Filters out tools with identical names and parameters
- Final deduplication check confirms zero duplicates

## 📁 Generated Benchmarks

The following benchmark files are available:

| File | Tool Count | Size | Description |
|------|------------|------|-------------|
| `BFCL_v3_tool_scaling_5.json` | 5 | 1.1 MB | Minimal tool set for basic testing |
| `BFCL_v3_tool_scaling_10.json` | 10 | 2.2 MB | Small tool set for quick evaluation |
| `BFCL_v3_tool_scaling_20.json` | 20 | 4.3 MB | Medium tool set for moderate complexity |
| `BFCL_v3_tool_scaling_50.json` | 50 | 11.2 MB | Large tool set for challenging scenarios |
| `BFCL_v3_tool_scaling_128.json` | 128 | 30.3 MB | Default comprehensive tool set |

Each benchmark contains **400 test cases** derived from `BFCL_v3_simple.json`.

## 🔧 Usage

### **Generate New Benchmarks**

```bash
# Generate all standard benchmarks
python generate_all_tool_scaling_benchmarks.py

# Generate custom benchmark
python -m bfcl_eval.scripts.tool_scaling_benchmark_efficient \
    --num_tools 64 \
    --output_file BFCL_v3_tool_scaling_64.json
```

### **Run Evaluation**

```bash
# Evaluate with tool scaling benchmark
python -m bfcl_eval.eval \
    --model gpt-4 \
    --test-category tool_scaling_128 \
    --num-threads 1
```

### **Integration with BFCL**

The tool scaling benchmarks are automatically integrated with the BFCL evaluation framework via `category_mapping.py`. Simply use the category names:

- `tool_scaling_5`
- `tool_scaling_10` 
- `tool_scaling_20`
- `tool_scaling_50`
- `tool_scaling_128`

## 🏗️ Architecture

### **Tool Aggregation Pipeline**
1. **Discovery**: Scans all Gorilla datasets for tool definitions
2. **Parsing**: Handles multiple tool formats (BFCL, APIZoo, APIBench, etc.)
3. **Deduplication**: Removes exact duplicates using JSON hashing
4. **Validation**: Ensures all tools have required fields

### **Embedding Computation**
1. **Text Conversion**: Converts tools to descriptive text format
2. **Parallel Processing**: Uses ThreadPoolExecutor with 8 workers
3. **Batch API Calls**: Processes 200 tools per API request
4. **Caching**: Stores embeddings to avoid recomputation
5. **Error Handling**: Fallback mechanisms for failed requests

### **Benchmark Generation**
1. **Matrix Pre-computation**: Creates embeddings matrix for all tools
2. **Vectorized Similarity**: Computes cosine similarity for all tools at once
3. **Top-K Selection**: Selects most similar tools for each query
4. **Format Generation**: Creates BFCL-compatible JSON format

## 📈 Performance Metrics

- **Tool Processing**: 33,090 tools in 70 seconds
- **Query Processing**: ~5 queries per second
- **Memory Efficiency**: Streaming processing for large datasets
- **Cache Hit Rate**: 100% after initial computation
- **Total Generation Time**: 7.5 minutes for all 5 benchmarks

## 🔍 Quality Validation

### **Semantic Relevance Examples**

**Query**: "Find the area of a triangle"
**Selected Tools**: 
- `calculate_triangle_area`
- `geometry.area_triangle` 
- `calc_area_triangle`
- `calculate_area`
- `math.triangle_area_heron`

**Query**: "Calculate the factorial of 5"
**Selected Tools**:
- `math_factorial`
- `math.factorial` (multiple variants)
- All semantically relevant to factorial computation

### **Tool Diversity**
- Tools sourced from 6 different datasets
- Multiple API styles and parameter formats
- Comprehensive coverage of mathematical, utility, and domain-specific functions

## 🚀 Future Enhancements

- **Dynamic Tool Counts**: Support for arbitrary tool counts
- **Domain-Specific Benchmarks**: Filter tools by category
- **Difficulty Levels**: Graduated complexity based on tool similarity
- **Multi-Modal Tools**: Support for tools with different input types
- **Real-Time Updates**: Automatic benchmark regeneration with new tools

## 📝 Technical Details

### **Dependencies**
- `openai` - For embedding computation
- `numpy` - For numerical operations
- `scikit-learn` - For cosine similarity
- `tqdm` - For progress tracking
- `concurrent.futures` - For parallel processing

### **Configuration**
- **Batch Size**: 200 (configurable)
- **Max Workers**: 8 (configurable)
- **Embedding Model**: `text-embedding-3-large`
- **Similarity Metric**: Cosine similarity
- **Cache Format**: Pickle files

### **File Structure**
```
bfcl_eval/
├── scripts/
│   └── tool_scaling_benchmark_efficient.py  # Main implementation
├── data/
│   ├── BFCL_v3_tool_scaling_*.json         # Generated benchmarks
│   └── embeddings_cache/                   # Cached embeddings (gitignored)
└── category_mapping.py                     # BFCL integration
```

## 🤝 Contributing

To add new tool sources or improve the benchmark:

1. **Add Tool Source**: Extend `aggregate_all_tools()` method
2. **Handle New Formats**: Update `_tool_to_text()` method  
3. **Test Integration**: Verify with BFCL evaluation framework
4. **Update Documentation**: Add details about new features

## 📄 License

This benchmark follows the same license as the Gorilla project.