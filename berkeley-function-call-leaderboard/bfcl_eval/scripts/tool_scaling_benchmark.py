#!/usr/bin/env python3
"""
Tool Scaling Benchmark Generator

This script creates a new benchmark that:
1. Aggregates ALL tools from Berkeley Function Calling Leaderboard
2. Uses OpenAI text-embedding-3-large to compute embeddings for each tool
3. For each query in BFCL_v3_simple.json, finds the top-k most similar tools
4. Generates a new benchmark file with configurable tool count

Usage:
    python tool_scaling_benchmark.py --num_tools 128 --output_file BFCL_v3_tool_scaling.json
"""

import json
import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from tqdm import tqdm
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolScalingBenchmark:
    def __init__(self, data_dir: str, cache_dir: str = None):
        """
        Initialize the Tool Scaling Benchmark generator.
        
        Args:
            data_dir: Path to the BFCL data directory
            cache_dir: Path to store embedding cache (default: data_dir/embeddings_cache)
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "embeddings_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
        
        # Cache for embeddings
        self.tool_embeddings_cache = {}
        self.query_embeddings_cache = {}
        
        # Load existing caches
        self._load_caches()
        
    def _load_caches(self):
        """Load existing embedding caches from disk."""
        tool_cache_file = self.cache_dir / "tool_embeddings.pkl"
        query_cache_file = self.cache_dir / "query_embeddings.pkl"
        
        if tool_cache_file.exists():
            with open(tool_cache_file, 'rb') as f:
                self.tool_embeddings_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.tool_embeddings_cache)} tool embeddings from cache")
        
        if query_cache_file.exists():
            with open(query_cache_file, 'rb') as f:
                self.query_embeddings_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.query_embeddings_cache)} query embeddings from cache")
    
    def _save_caches(self):
        """Save embedding caches to disk."""
        tool_cache_file = self.cache_dir / "tool_embeddings.pkl"
        query_cache_file = self.cache_dir / "query_embeddings.pkl"
        
        with open(tool_cache_file, 'wb') as f:
            pickle.dump(self.tool_embeddings_cache, f)
        
        with open(query_cache_file, 'wb') as f:
            pickle.dump(self.query_embeddings_cache, f)
        
        logger.info("Saved embedding caches to disk")
    
    def _get_tool_hash(self, tool: Dict[str, Any]) -> str:
        """Generate a hash for a tool to use as cache key."""
        tool_str = json.dumps(tool, sort_keys=True)
        return hashlib.md5(tool_str.encode()).hexdigest()
    
    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for a query to use as cache key."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_embedding(self, text: str, cache_dict: Dict[str, np.ndarray], cache_key: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        if cache_key in cache_dict:
            return cache_dict[cache_key]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            cache_dict[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            raise
    
    def _tool_to_text(self, tool: Dict[str, Any]) -> str:
        """Convert a tool definition to text for embedding."""
        name = tool.get('name', '')
        description = tool.get('description', '')
        parameters = tool.get('parameters', {})
        
        # Create a comprehensive text representation
        text_parts = [f"Function: {name}"]
        
        if description:
            text_parts.append(f"Description: {description}")
        
        if parameters and 'properties' in parameters:
            text_parts.append("Parameters:")
            for param_name, param_info in parameters['properties'].items():
                param_desc = param_info.get('description', '')
                param_type = param_info.get('type', '')
                text_parts.append(f"- {param_name} ({param_type}): {param_desc}")
        
        return " ".join(text_parts)
    
    def aggregate_all_tools(self) -> List[Dict[str, Any]]:
        """Aggregate all tools from all BFCL test files."""
        all_tools = []
        seen_tools = set()
        
        # Get all JSON files in the data directory
        json_files = list(self.data_dir.glob("BFCL_v3_*.json"))
        
        logger.info(f"Found {len(json_files)} BFCL test files")
        
        for json_file in json_files:
            logger.info(f"Processing {json_file.name}")
            
            with open(json_file, 'r') as f:
                for line in f:
                    try:
                        test_case = json.loads(line.strip())
                        functions = test_case.get('function', [])
                        
                        for func in functions:
                            # Create a unique identifier for the tool
                            tool_hash = self._get_tool_hash(func)
                            
                            if tool_hash not in seen_tools:
                                seen_tools.add(tool_hash)
                                all_tools.append(func)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Aggregated {len(all_tools)} unique tools")
        return all_tools
    
    def compute_tool_embeddings(self, tools: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Compute embeddings for all tools."""
        tool_embeddings = {}
        
        logger.info("Computing tool embeddings...")
        for tool in tqdm(tools, desc="Computing tool embeddings"):
            tool_hash = self._get_tool_hash(tool)
            tool_text = self._tool_to_text(tool)
            
            embedding = self._get_embedding(tool_text, self.tool_embeddings_cache, tool_hash)
            tool_embeddings[tool_hash] = embedding
        
        return tool_embeddings
    
    def load_simple_queries(self) -> List[Dict[str, Any]]:
        """Load queries from BFCL_v3_simple.json."""
        simple_file = self.data_dir / "BFCL_v3_simple.json"
        queries = []
        
        with open(simple_file, 'r') as f:
            for line in f:
                try:
                    test_case = json.loads(line.strip())
                    queries.append(test_case)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(queries)} queries from BFCL_v3_simple.json")
        return queries
    
    def compute_query_embeddings(self, queries: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Compute embeddings for all queries."""
        query_embeddings = {}
        
        logger.info("Computing query embeddings...")
        for query_data in tqdm(queries, desc="Computing query embeddings"):
            query_id = query_data['id']
            
            # Extract the user query text
            question = query_data['question'][0][0]['content']
            query_hash = self._get_query_hash(question)
            
            embedding = self._get_embedding(question, self.query_embeddings_cache, query_hash)
            query_embeddings[query_id] = embedding
        
        return query_embeddings
    
    def find_top_k_tools(self, query_embedding: np.ndarray, tool_embeddings: Dict[str, np.ndarray], 
                        tools: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Find top-k most similar tools for a query."""
        # Prepare tool embeddings matrix
        tool_hashes = []
        embeddings_matrix = []
        
        for tool in tools:
            tool_hash = self._get_tool_hash(tool)
            if tool_hash in tool_embeddings:
                tool_hashes.append(tool_hash)
                embeddings_matrix.append(tool_embeddings[tool_hash])
        
        if not embeddings_matrix:
            return tools[:k]  # Fallback if no embeddings
        
        # Convert to numpy array for efficient computation
        embeddings_matrix = np.array(embeddings_matrix)
        
        # Compute cosine similarities all at once
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            embeddings_matrix
        )[0]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top-k tools
        top_k_tools = []
        tool_hash_to_tool = {self._get_tool_hash(tool): tool for tool in tools}
        
        for i in sorted_indices[:k]:
            tool_hash = tool_hashes[i]
            top_k_tools.append(tool_hash_to_tool[tool_hash])
        
        return top_k_tools
    
    def generate_benchmark(self, num_tools: int = 128, output_file: str = None) -> str:
        """Generate the tool scaling benchmark."""
        if output_file is None:
            output_file = self.data_dir / f"BFCL_v3_tool_scaling_{num_tools}.json"
        else:
            output_file = Path(output_file)
        
        logger.info(f"Generating tool scaling benchmark with {num_tools} tools")
        
        # Step 1: Aggregate all tools
        all_tools = self.aggregate_all_tools()
        
        # Step 2: Compute tool embeddings
        tool_embeddings = self.compute_tool_embeddings(all_tools)
        
        # Step 3: Load simple queries
        queries = self.load_simple_queries()
        
        # Step 4: Compute query embeddings
        query_embeddings = self.compute_query_embeddings(queries)
        
        # Step 5: Generate new benchmark
        logger.info("Generating new benchmark file...")
        with open(output_file, 'w') as f:
            for query_data in tqdm(queries, desc="Processing queries"):
                query_id = query_data['id']
                query_embedding = query_embeddings[query_id]
                
                # Find top-k tools for this query
                top_k_tools = self.find_top_k_tools(query_embedding, tool_embeddings, all_tools, num_tools)
                
                # Create new test case
                new_test_case = {
                    'id': f"tool_scaling_{num_tools}_{query_data['id']}",
                    'question': query_data['question'],
                    'function': top_k_tools
                }
                
                f.write(json.dumps(new_test_case) + '\n')
        
        # Save caches
        self._save_caches()
        
        logger.info(f"Generated benchmark saved to {output_file}")
        return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Generate Tool Scaling Benchmark")
    parser.add_argument("--data_dir", default="./bfcl_eval/data", 
                       help="Path to BFCL data directory")
    parser.add_argument("--num_tools", type=int, default=128,
                       help="Number of top tools to include (default: 128)")
    parser.add_argument("--output_file", default=None,
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--cache_dir", default=None,
                       help="Cache directory for embeddings (default: data_dir/embeddings_cache)")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Create benchmark generator
    benchmark = ToolScalingBenchmark(args.data_dir, args.cache_dir)
    
    # Generate benchmark
    output_file = benchmark.generate_benchmark(args.num_tools, args.output_file)
    
    print(f"Tool scaling benchmark generated: {output_file}")


if __name__ == "__main__":
    main()