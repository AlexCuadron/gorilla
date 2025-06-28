#!/usr/bin/env python3
"""
Tool Scaling Benchmark Generator (Efficient Version)

This script creates a new benchmark that:
1. Aggregates ALL tools from Berkeley Function Calling Leaderboard
2. Uses OpenAI text-embedding-3-large to compute embeddings for each tool (batched)
3. For each query in BFCL_v3_simple.json, finds the top-k most similar tools
4. Generates a new benchmark file with configurable tool count

Usage:
    python tool_scaling_benchmark_efficient.py --num_tools 128 --output_file BFCL_v3_tool_scaling.json
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
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolScalingBenchmarkEfficient:
    def __init__(self, data_dir: str, cache_dir: str = None, batch_size: int = 100):
        """
        Initialize the Tool Scaling Benchmark generator.
        
        Args:
            data_dir: Path to the BFCL data directory
            cache_dir: Path to store embedding cache (default: data_dir/embeddings_cache)
            batch_size: Number of texts to process in each API call
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "embeddings_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
        
        # Cache for embeddings
        self.tool_embeddings_cache = {}
        self.query_embeddings_cache = {}
        
        # Load existing caches
        self._load_caches()
        
    def _load_caches(self):
        """Load existing embedding caches from disk."""
        tool_cache_file = self.cache_dir / "tool_embeddings_efficient.pkl"
        query_cache_file = self.cache_dir / "query_embeddings_efficient.pkl"
        
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
        tool_cache_file = self.cache_dir / "tool_embeddings_efficient.pkl"
        query_cache_file = self.cache_dir / "query_embeddings_efficient.pkl"
        
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
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=texts
            )
            embeddings = [np.array(data.embedding) for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings for batch: {e}")
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
        
        # Get all JSON files in the data directory (exclude our generated ones)
        json_files = [f for f in self.data_dir.glob("BFCL_v3_*.json") 
                     if not f.name.startswith("BFCL_v3_tool_scaling")]
        
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
        """Compute embeddings for all tools using batched API calls."""
        tool_embeddings = {}
        
        # Prepare data for batching
        tools_to_process = []
        tool_hashes_to_process = []
        tool_texts_to_process = []
        
        for tool in tools:
            tool_hash = self._get_tool_hash(tool)
            
            if tool_hash in self.tool_embeddings_cache:
                tool_embeddings[tool_hash] = self.tool_embeddings_cache[tool_hash]
            else:
                tools_to_process.append(tool)
                tool_hashes_to_process.append(tool_hash)
                tool_texts_to_process.append(self._tool_to_text(tool))
        
        logger.info(f"Computing embeddings for {len(tools_to_process)} new tools")
        
        # Process in batches
        for i in tqdm(range(0, len(tool_texts_to_process), self.batch_size), 
                     desc="Computing tool embeddings"):
            batch_texts = tool_texts_to_process[i:i+self.batch_size]
            batch_hashes = tool_hashes_to_process[i:i+self.batch_size]
            
            if batch_texts:
                try:
                    batch_embeddings = self._get_embeddings_batch(batch_texts)
                    
                    for hash_key, embedding in zip(batch_hashes, batch_embeddings):
                        tool_embeddings[hash_key] = embedding
                        self.tool_embeddings_cache[hash_key] = embedding
                    
                    # Add a small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//self.batch_size}: {e}")
                    # Fall back to individual processing for this batch
                    for j, (text, hash_key) in enumerate(zip(batch_texts, batch_hashes)):
                        try:
                            embedding = self._get_embeddings_batch([text])[0]
                            tool_embeddings[hash_key] = embedding
                            self.tool_embeddings_cache[hash_key] = embedding
                            time.sleep(0.1)
                        except Exception as e2:
                            logger.error(f"Error processing individual tool {hash_key}: {e2}")
        
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
        """Compute embeddings for all queries using batched API calls."""
        query_embeddings = {}
        
        # Prepare data for batching
        queries_to_process = []
        query_ids_to_process = []
        query_texts_to_process = []
        query_hashes_to_process = []
        
        for query_data in queries:
            query_id = query_data['id']
            question = query_data['question'][0][0]['content']
            query_hash = self._get_query_hash(question)
            
            if query_hash in self.query_embeddings_cache:
                query_embeddings[query_id] = self.query_embeddings_cache[query_hash]
            else:
                queries_to_process.append(query_data)
                query_ids_to_process.append(query_id)
                query_texts_to_process.append(question)
                query_hashes_to_process.append(query_hash)
        
        logger.info(f"Computing embeddings for {len(queries_to_process)} new queries")
        
        # Process in batches
        for i in tqdm(range(0, len(query_texts_to_process), self.batch_size), 
                     desc="Computing query embeddings"):
            batch_texts = query_texts_to_process[i:i+self.batch_size]
            batch_ids = query_ids_to_process[i:i+self.batch_size]
            batch_hashes = query_hashes_to_process[i:i+self.batch_size]
            
            if batch_texts:
                try:
                    batch_embeddings = self._get_embeddings_batch(batch_texts)
                    
                    for query_id, hash_key, embedding in zip(batch_ids, batch_hashes, batch_embeddings):
                        query_embeddings[query_id] = embedding
                        self.query_embeddings_cache[hash_key] = embedding
                    
                    # Add a small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error processing query batch {i//self.batch_size}: {e}")
                    # Fall back to individual processing for this batch
                    for j, (text, query_id, hash_key) in enumerate(zip(batch_texts, batch_ids, batch_hashes)):
                        try:
                            embedding = self._get_embeddings_batch([text])[0]
                            query_embeddings[query_id] = embedding
                            self.query_embeddings_cache[hash_key] = embedding
                            time.sleep(0.1)
                        except Exception as e2:
                            logger.error(f"Error processing individual query {query_id}: {e2}")
        
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
    parser = argparse.ArgumentParser(description="Generate Tool Scaling Benchmark (Efficient Version)")
    parser.add_argument("--data_dir", default="./bfcl_eval/data", 
                       help="Path to BFCL data directory")
    parser.add_argument("--num_tools", type=int, default=128,
                       help="Number of top tools to include (default: 128)")
    parser.add_argument("--output_file", default=None,
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--cache_dir", default=None,
                       help="Cache directory for embeddings (default: data_dir/embeddings_cache)")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for API calls (default: 100)")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Create benchmark generator
    benchmark = ToolScalingBenchmarkEfficient(args.data_dir, args.cache_dir, args.batch_size)
    
    # Generate benchmark
    output_file = benchmark.generate_benchmark(args.num_tools, args.output_file)
    
    print(f"Tool scaling benchmark generated: {output_file}")


if __name__ == "__main__":
    main()