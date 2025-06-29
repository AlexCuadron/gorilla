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
import concurrent.futures
import threading
import time
from functools import partial

class ToolScalingBenchmarkEfficient:
    def __init__(self, data_dir: str, cache_dir: str = None, batch_size: int = 200):
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
            print(f"📦 Loaded {len(self.tool_embeddings_cache)} tool embeddings from cache")
        
        if query_cache_file.exists():
            with open(query_cache_file, 'rb') as f:
                self.query_embeddings_cache = pickle.load(f)
            print(f"📦 Loaded {len(self.query_embeddings_cache)} query embeddings from cache")
    
    def _save_caches(self):
        """Save embedding caches to disk."""
        tool_cache_file = self.cache_dir / "tool_embeddings_efficient.pkl"
        query_cache_file = self.cache_dir / "query_embeddings_efficient.pkl"
        
        with open(tool_cache_file, 'wb') as f:
            pickle.dump(self.tool_embeddings_cache, f)
        
        with open(query_cache_file, 'wb') as f:
            pickle.dump(self.query_embeddings_cache, f)
        
        print("💾 Saved embedding caches to disk")
    
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
            print(f"❗️ Error getting embeddings for batch: {e}")
            raise
    
    def _tool_to_text(self, tool: Dict[str, Any]) -> str:
        """Convert a tool definition to text for embedding, handling multiple formats."""
        text_parts = []
        
        # Handle different tool formats from various datasets
        
        # 1. BFCL format: {name, description, parameters}
        if 'name' in tool and 'description' in tool:
            name = tool.get('name', '')
            description = tool.get('description', '')
            parameters = tool.get('parameters', {})
            
            text_parts.append(f"Function: {name}")
            if description:
                text_parts.append(f"Description: {description}")
            
            if parameters and 'properties' in parameters:
                text_parts.append("Parameters:")
                for param_name, param_info in parameters['properties'].items():
                    param_desc = param_info.get('description', '')
                    param_type = param_info.get('type', '')
                    text_parts.append(f"- {param_name} ({param_type}): {param_desc}")
        
        # 2. APIZoo format: {api_name, api_call, functionality, meta_data}
        elif 'api_name' in tool:
            api_name = tool.get('api_name', '')
            api_call = tool.get('api_call', '')
            functionality = tool.get('functionality', '')
            meta_data = tool.get('meta_data', {})
            
            text_parts.append(f"API: {api_name}")
            if api_call:
                text_parts.append(f"Call: {api_call}")
            if functionality:
                text_parts.append(f"Functionality: {functionality}")
            if isinstance(meta_data, dict) and 'description' in meta_data:
                text_parts.append(f"Description: {meta_data['description']}")
        
        # 3. APIBench format: {api_call, api_data, provider}
        elif 'api_call' in tool and 'api_data' in tool:
            api_call = tool.get('api_call', '')
            api_data = tool.get('api_data', {})
            provider = tool.get('provider', '')
            
            text_parts.append(f"API Call: {api_call}")
            if provider:
                text_parts.append(f"Provider: {provider}")
            
            if isinstance(api_data, dict):
                domain = api_data.get('domain', '')
                functionality = api_data.get('functionality', '')
                description = api_data.get('description', '')
                
                if domain:
                    text_parts.append(f"Domain: {domain}")
                if functionality:
                    text_parts.append(f"Functionality: {functionality}")
                if description:
                    text_parts.append(f"Description: {description}")
        
        # 4. OpenFunctions format: {name, api_call, description, parameters}
        elif 'api_call' in tool:
            name = tool.get('name', '')
            api_call = tool.get('api_call', '')
            description = tool.get('description', '')
            parameters = tool.get('parameters', {})
            
            if name:
                text_parts.append(f"Function: {name}")
            text_parts.append(f"API Call: {api_call}")
            if description:
                text_parts.append(f"Description: {description}")
            
            if isinstance(parameters, dict) and 'properties' in parameters:
                text_parts.append("Parameters:")
                for param_name, param_info in parameters['properties'].items():
                    param_desc = param_info.get('description', '')
                    param_type = param_info.get('type', '')
                    text_parts.append(f"- {param_name} ({param_type}): {param_desc}")
        
        # 5. Generic fallback - try to extract any useful text
        else:
            # Look for common fields that might contain useful information
            for key in ['name', 'api_name', 'function_name', 'title']:
                if key in tool and tool[key]:
                    text_parts.append(f"Name: {tool[key]}")
                    break
            
            for key in ['description', 'summary', 'functionality', 'purpose']:
                if key in tool and tool[key]:
                    text_parts.append(f"Description: {tool[key]}")
                    break
            
            for key in ['api_call', 'call', 'usage', 'example']:
                if key in tool and tool[key]:
                    text_parts.append(f"Usage: {tool[key]}")
                    break
        
        # If we couldn't extract anything meaningful, use the entire tool as text
        if not text_parts:
            text_parts.append(str(tool))
        
        return " ".join(text_parts)
    
    def _normalize_tool_to_bfcl_format(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a tool from any format to standard BFCL format."""
        normalized_tool = {}
        
        # 1. BFCL format: {name, description, parameters} - already correct
        if ('name' in tool and 'description' in tool and 
            'parameters' in tool and isinstance(tool.get('parameters'), dict)):
            return tool.copy()
        
        # 2. APIZoo/RapidAPI format: {api_name, description, parameters} or required/optional format
        elif ('api_name' in tool or 
              ('name' in tool and 'parameters' in tool and isinstance(tool.get('parameters'), list)) or
              ('name' in tool and 'parameters' in tool and isinstance(tool.get('parameters'), dict) and 
               ('required' in tool.get('parameters', {}) or 'optional' in tool.get('parameters', {})))):
            normalized_tool['name'] = tool.get('api_name', tool.get('name', 'unknown_function'))
            normalized_tool['description'] = tool.get('description', tool.get('functionality', ''))
            
            # Handle parameters - might be list or dict format
            params = tool.get('parameters', [])
            if isinstance(params, list):
                # Convert list format to dict format
                properties = {}
                required = []
                for param in params:
                    if isinstance(param, dict) and 'name' in param:
                        param_name = param['name']
                        properties[param_name] = {
                            'type': param.get('type', 'string').lower(),
                            'description': param.get('description', '')
                        }
                        # Assume all parameters are required if not specified
                        required.append(param_name)
                
                normalized_tool['parameters'] = {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            elif isinstance(params, dict) and ('required' in params or 'optional' in params):
                # Handle required/optional format
                properties = {}
                required = []
                
                # Process required parameters
                for param in params.get('required', []):
                    if isinstance(param, dict) and 'name' in param:
                        param_name = param['name']
                        properties[param_name] = {
                            'type': param.get('type', 'string').lower(),
                            'description': param.get('description', '')
                        }
                        required.append(param_name)
                
                # Process optional parameters
                for param in params.get('optional', []):
                    if isinstance(param, dict) and 'name' in param and param['name'] != 'N/A':
                        param_name = param['name']
                        properties[param_name] = {
                            'type': param.get('type', 'string').lower(),
                            'description': param.get('description', '')
                        }
                        # Don't add to required list
                
                normalized_tool['parameters'] = {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            else:
                normalized_tool['parameters'] = params
        
        # 3. Hugging Face/API format: {api_call, provider, api_data}
        elif 'api_call' in tool and 'api_data' in tool:
            api_data = tool.get('api_data', {})
            normalized_tool['name'] = api_data.get('api_name', tool.get('provider', 'unknown_function'))
            normalized_tool['description'] = api_data.get('description', tool.get('code', ''))
            
            # Extract parameters from api_arguments if available
            api_args = api_data.get('api_arguments', [])
            if isinstance(api_args, list):
                properties = {}
                required = []
                for arg in api_args:
                    if isinstance(arg, str):
                        properties[arg] = {
                            'type': 'string',
                            'description': f'Parameter for {arg}'
                        }
                        required.append(arg)
                
                normalized_tool['parameters'] = {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            elif isinstance(api_args, dict):
                # Convert dict format
                properties = {}
                required = []
                for key, value in api_args.items():
                    properties[key] = {
                        'type': 'string' if not isinstance(value, bool) else 'boolean',
                        'description': f'Parameter for {key}'
                    }
                    required.append(key)
                
                normalized_tool['parameters'] = {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            else:
                normalized_tool['parameters'] = {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
        
        # 4. Generic fallback
        else:
            # Try to extract name from various fields
            name = None
            for key in ['name', 'api_name', 'function_name', 'title']:
                if key in tool and tool[key]:
                    name = tool[key]
                    break
            
            # Try to extract description from various fields
            description = None
            for key in ['description', 'summary', 'functionality', 'purpose']:
                if key in tool and tool[key]:
                    description = tool[key]
                    break
            
            normalized_tool['name'] = name or 'unknown_function'
            normalized_tool['description'] = description or str(tool)
            normalized_tool['parameters'] = {
                'type': 'object',
                'properties': {},
                'required': []
            }
        
        # Ensure all required fields are present
        if 'name' not in normalized_tool:
            normalized_tool['name'] = 'unknown_function'
        if 'description' not in normalized_tool:
            normalized_tool['description'] = ''
        if 'parameters' not in normalized_tool:
            normalized_tool['parameters'] = {
                'type': 'object',
                'properties': {},
                'required': []
            }
        
        return normalized_tool
    
    def aggregate_all_tools(self) -> List[Dict[str, Any]]:
        """Aggregate all tools from ALL Gorilla datasets."""
        import ast
        from pathlib import Path
        
        all_tools = []
        seen_tools = set()
        
        print(f"🔍 Aggregating tools from ALL Gorilla datasets...")
        
        # 1. BFCL files from berkeley-function-call-leaderboard
        bfcl_files = []
        for pattern in ['**/*.json']:
            files = list(self.data_dir.glob(pattern))
            bfcl_files.extend([f for f in files if 'BFCL_v3_' in f.name and not f.name.startswith('BFCL_v3_tool_scaling')])
        
        print(f"📄 Processing {len(bfcl_files)} BFCL files...")
        bfcl_tools = 0
        for json_file in bfcl_files:
            try:
                with open(json_file, 'r') as f:
                    for line in f:
                        try:
                            test_case = json.loads(line.strip())
                            functions = test_case.get('function', [])
                            
                            for func in functions:
                                tool_hash = self._get_tool_hash(func)
                                if tool_hash not in seen_tools:
                                    seen_tools.add(tool_hash)
                                    all_tools.append(func)
                                    bfcl_tools += 1
                        except:
                            continue
            except Exception as e:
                print(f"❗️ Error processing {json_file}: {e}")
        
        # 2. Multi-turn func doc files
        func_doc_dir = self.data_dir / 'multi_turn_func_doc'
        func_doc_tools = 0
        if func_doc_dir.exists():
            func_doc_files = list(func_doc_dir.glob('*.json'))
            print(f"📄 Processing {len(func_doc_files)} func_doc files...")
            for json_file in func_doc_files:
                try:
                    with open(json_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    tool = json.loads(line)
                                    if 'name' in tool:
                                        tool_hash = self._get_tool_hash(tool)
                                        if tool_hash not in seen_tools:
                                            seen_tools.add(tool_hash)
                                            all_tools.append(tool)
                                            func_doc_tools += 1
                                except:
                                    pass
                except Exception as e:
                    print(f"❗️ Error processing {json_file}: {e}")
        
        # 3. OpenFunctions datasets (if available)
        gorilla_root = self.data_dir.absolute().parent.parent.parent  # Go up to gorilla root
        openfunctions_dir = gorilla_root / 'openfunctions' / 'openfunctions-v1'
        openfunctions_tools = 0
        if openfunctions_dir.exists():
            print(f"📄 Processing OpenFunctions datasets...")
            
            # Test file
            test_file = openfunctions_dir / 'gorilla_openfunctions_v1_test.json'
            if test_file.exists():
                try:
                    with open(test_file, 'r') as f:
                        data = json.load(f)
                        for entry in data:
                            if 'function' in entry:
                                func = entry['function']
                                tool_hash = self._get_tool_hash(func)
                                if tool_hash not in seen_tools:
                                    seen_tools.add(tool_hash)
                                    all_tools.append(func)
                                    openfunctions_tools += 1
                except Exception as e:
                    print(f"❗️ Error processing OpenFunctions test: {e}")
            
            # Train file (JSONL format)
            train_file = openfunctions_dir / 'gorilla_openfunctions_v1_train.json'
            if train_file.exists():
                try:
                    with open(train_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'Functions' in data:
                                        functions = data['Functions']
                                        for func_str in functions:
                                            try:
                                                func = ast.literal_eval(func_str)
                                                tool_hash = self._get_tool_hash(func)
                                                if tool_hash not in seen_tools:
                                                    seen_tools.add(tool_hash)
                                                    all_tools.append(func)
                                                    openfunctions_tools += 1
                                            except:
                                                pass
                                except:
                                    pass
                except Exception as e:
                    print(f"❗️ Error processing OpenFunctions train: {e}")
        
        # 4. APIZoo datasets
        apizoo_dir = gorilla_root / 'data' / 'apizoo'
        apizoo_tools = 0
        if apizoo_dir.exists():
            apizoo_files = list(apizoo_dir.glob('*.json')) + list(apizoo_dir.glob('*.JSON'))
            print(f"📄 Processing {len(apizoo_files)} APIZoo files...")
            for json_file in apizoo_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if 'api_name' in item or 'api_call' in item:
                                    tool_hash = self._get_tool_hash(item)
                                    if tool_hash not in seen_tools:
                                        seen_tools.add(tool_hash)
                                        all_tools.append(item)
                                        apizoo_tools += 1
                        elif isinstance(data, dict):
                            if 'api_name' in data or 'api_call' in data:
                                tool_hash = self._get_tool_hash(data)
                                if tool_hash not in seen_tools:
                                    seen_tools.add(tool_hash)
                                    all_tools.append(data)
                                    apizoo_tools += 1
                except Exception as e:
                    # Some files have JSON syntax errors, skip them
                    pass
        
        # 5. APIBench datasets (JSONL format)
        apibench_dir = gorilla_root / 'data' / 'apibench'
        apibench_tools = 0
        if apibench_dir.exists():
            apibench_files = list(apibench_dir.glob('*.json'))
            print(f"📄 Processing {len(apibench_files)} APIBench files...")
            for json_file in apibench_files:
                try:
                    with open(json_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    if any(key in item for key in ['api_call', 'api_data', 'provider']):
                                        tool_hash = self._get_tool_hash(item)
                                        if tool_hash not in seen_tools:
                                            seen_tools.add(tool_hash)
                                            all_tools.append(item)
                                            apibench_tools += 1
                                except:
                                    pass
                except Exception as e:
                    print(f"❗️ Error processing {json_file}: {e}")
        
        # 6. API directory (JSONL format)
        api_dir = gorilla_root / 'data' / 'api'
        api_tools = 0
        if api_dir.exists():
            api_files = list(api_dir.glob('*.jsonl'))
            print(f"📄 Processing {len(api_files)} API files...")
            for jsonl_file in api_files:
                try:
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    if any(key in item for key in ['api_call', 'api_name', 'name', 'function']):
                                        tool_hash = self._get_tool_hash(item)
                                        if tool_hash not in seen_tools:
                                            seen_tools.add(tool_hash)
                                            all_tools.append(item)
                                            api_tools += 1
                                except:
                                    pass
                except Exception as e:
                    print(f"❗️ Error processing {jsonl_file}: {e}")
        
        print(f"🎉 Total unique tools aggregated: {len(all_tools):,}")
        print(f"📊 Breakdown:")
        print(f"   BFCL files: {bfcl_tools:,}")
        print(f"   Func doc: {func_doc_tools:,}")
        print(f"   OpenFunctions: {openfunctions_tools:,}")
        print(f"   APIZoo: {apizoo_tools:,}")
        print(f"   APIBench: {apibench_tools:,}")
        print(f"   API directory: {api_tools:,}")
        
        # Final deduplication check to ensure absolutely no duplicates
        print(f"🔍 Performing final deduplication check...")
        final_tools = []
        final_seen_hashes = set()
        
        for tool in all_tools:
            tool_hash = self._get_tool_hash(tool)
            if tool_hash not in final_seen_hashes:
                final_seen_hashes.add(tool_hash)
                final_tools.append(tool)
        
        removed_duplicates = len(all_tools) - len(final_tools)
        if removed_duplicates > 0:
            print(f"🧹 Removed {removed_duplicates} additional duplicate tools")
        
        print(f"✅ Final unique tool count: {len(final_tools):,}")
        return final_tools
    
    def _process_batch_parallel(self, batch_data: Tuple[List[str], List[str], int]) -> Tuple[List[str], List[np.ndarray], int]:
        """Process a single batch of embeddings in parallel."""
        batch_texts, batch_hashes, batch_idx = batch_data
        
        try:
            batch_embeddings = self._get_embeddings_batch(batch_texts)
            return batch_hashes, batch_embeddings, batch_idx
        except Exception as e:
            print(f"❗️ Error processing batch {batch_idx}: {e}")
            # Fall back to individual processing
            individual_embeddings = []
            for text in batch_texts:
                try:
                    embedding = self._get_embeddings_batch([text])[0]
                    individual_embeddings.append(embedding)
                    time.sleep(0.05)  # Shorter delay for individual requests
                except Exception as e2:
                    print(f"❗️ Error processing individual text: {e2}")
                    # Use zero embedding as fallback
                    individual_embeddings.append(np.zeros(3072))  # text-embedding-3-large dimension
            return batch_hashes, individual_embeddings, batch_idx

    def compute_tool_embeddings(self, tools: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Compute embeddings for all tools using parallel batched API calls."""
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
        
        print(f"🧮 Computing embeddings for {len(tools_to_process)} new tools")
        
        if not tools_to_process:
            return tool_embeddings
        
        # Prepare batches for parallel processing
        batches = []
        for i in range(0, len(tool_texts_to_process), self.batch_size):
            batch_texts = tool_texts_to_process[i:i+self.batch_size]
            batch_hashes = tool_hashes_to_process[i:i+self.batch_size]
            batches.append((batch_texts, batch_hashes, i // self.batch_size))
        
        print(f"🚀 Processing {len(batches)} batches in parallel with {min(8, len(batches))} workers")
        
        # Process batches in parallel with rate limiting
        max_workers = min(8, len(batches))  # Limit concurrent requests to avoid rate limits
        completed_batches = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(self._process_batch_parallel, batch): batch 
                             for batch in batches}
            
            # Process completed batches with progress bar
            with tqdm(total=len(batches), desc="Computing tool embeddings") as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_hashes, batch_embeddings, batch_idx = future.result()
                        
                        # Store embeddings
                        for hash_key, embedding in zip(batch_hashes, batch_embeddings):
                            tool_embeddings[hash_key] = embedding
                            self.tool_embeddings_cache[hash_key] = embedding
                        
                        completed_batches += 1
                        pbar.update(1)
                        
                        # Save cache periodically to avoid losing progress
                        if completed_batches % 10 == 0:
                            self._save_caches()
                        
                        # Rate limiting: small delay between batch completions
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"❗️ Error processing batch: {e}")
                        pbar.update(1)
        
        # Final cache save
        self._save_caches()
        print(f"✅ Completed embedding computation for {len(tools_to_process)} tools")
        
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
        
        print(f"📝 Loaded {len(queries)} queries from BFCL_v3_simple.json")
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
        
        print(f"🧮 Computing embeddings for {len(queries_to_process)} new queries")
        
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
                    print(f"❗️ Error processing query batch {i//self.batch_size}: {e}")
                    # Fall back to individual processing for this batch
                    for j, (text, query_id, hash_key) in enumerate(zip(batch_texts, batch_ids, batch_hashes)):
                        try:
                            embedding = self._get_embeddings_batch([text])[0]
                            query_embeddings[query_id] = embedding
                            self.query_embeddings_cache[hash_key] = embedding
                            time.sleep(0.1)
                        except Exception as e2:
                            print(f"❗️ Error processing individual query {query_id}: {e2}")
        
        return query_embeddings
    
    def find_top_k_tools(self, query_embedding: np.ndarray, tool_embeddings: Dict[str, np.ndarray], 
                        tools: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Find top-k most similar tools for a query."""
        # Prepare tool embeddings matrix and mappings efficiently
        tool_hashes = []
        embeddings_matrix = []
        hash_to_tool = {}
        
        for tool in tools:
            tool_hash = self._get_tool_hash(tool)
            if tool_hash in tool_embeddings:
                tool_hashes.append(tool_hash)
                embeddings_matrix.append(tool_embeddings[tool_hash])
                hash_to_tool[tool_hash] = tool
        
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
        
        # Get top-k tools using pre-computed mapping and normalize them
        top_k_tools = []
        for i in sorted_indices[:k]:
            tool_hash = tool_hashes[i]
            original_tool = hash_to_tool[tool_hash]
            normalized_tool = self._normalize_tool_to_bfcl_format(original_tool)
            top_k_tools.append(normalized_tool)
        
        return top_k_tools
    
    def _find_top_k_tools_optimized(self, query_embedding: np.ndarray, embeddings_matrix: np.ndarray,
                                   tool_hashes: List[str], hash_to_tool: Dict[str, Dict[str, Any]], 
                                   k: int) -> List[Dict[str, Any]]:
        """Optimized version that uses pre-computed embeddings matrix."""
        # Compute cosine similarities all at once
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            embeddings_matrix
        )[0]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top-k tools using pre-computed mapping and normalize them
        top_k_tools = []
        for i in sorted_indices[:k]:
            tool_hash = tool_hashes[i]
            original_tool = hash_to_tool[tool_hash]
            normalized_tool = self._normalize_tool_to_bfcl_format(original_tool)
            top_k_tools.append(normalized_tool)
        
        return top_k_tools
    
    def generate_benchmark(self, num_tools: int = 128, output_file: str = None) -> str:
        """Generate the tool scaling benchmark."""
        if output_file is None:
            output_file = self.data_dir / f"BFCL_v3_tool_scaling_{num_tools}.json"
        else:
            output_file = Path(output_file)
        
        print(f"🎯 Generating tool scaling benchmark with {num_tools} tools")
        
        # Step 1: Aggregate all tools
        all_tools = self.aggregate_all_tools()
        
        # Step 2: Compute tool embeddings
        tool_embeddings = self.compute_tool_embeddings(all_tools)
        
        # Step 3: Load simple queries
        queries = self.load_simple_queries()
        
        # Step 4: Compute query embeddings
        query_embeddings = self.compute_query_embeddings(queries)
        
        # Step 5: Pre-compute tool embeddings matrix for efficiency
        print("🔧 Pre-computing tool embeddings matrix...")
        tool_hashes = []
        embeddings_matrix = []
        hash_to_tool = {}
        
        for tool in all_tools:
            tool_hash = self._get_tool_hash(tool)
            if tool_hash in tool_embeddings:
                tool_hashes.append(tool_hash)
                embeddings_matrix.append(tool_embeddings[tool_hash])
                hash_to_tool[tool_hash] = tool
        
        embeddings_matrix = np.array(embeddings_matrix)
        print(f"📊 Pre-computed matrix shape: {embeddings_matrix.shape}")
        
        # Step 6: Generate new benchmark
        print("📝 Generating new benchmark file...")
        with open(output_file, 'w') as f:
            for query_data in tqdm(queries, desc="Processing queries"):
                query_id = query_data['id']
                query_embedding = query_embeddings[query_id]
                
                # Find top-k tools for this query using pre-computed matrix
                top_k_tools = self._find_top_k_tools_optimized(
                    query_embedding, embeddings_matrix, tool_hashes, hash_to_tool, num_tools
                )
                
                # Create new test case
                new_test_case = {
                    'id': f"tool_scaling_{num_tools}_{query_data['id']}",
                    'question': query_data['question'],
                    'function': top_k_tools
                }
                
                f.write(json.dumps(new_test_case) + '\n')
        
        # Save caches
        self._save_caches()
        
        print(f"✅ Generated benchmark saved to {output_file}")
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
        print("❗️ OPENAI_API_KEY environment variable not set")
        return
    
    # Create benchmark generator
    benchmark = ToolScalingBenchmarkEfficient(args.data_dir, args.cache_dir, args.batch_size)
    
    # Generate benchmark
    output_file = benchmark.generate_benchmark(args.num_tools, args.output_file)
    
    print(f"🎉 Tool scaling benchmark generated: {output_file}")


if __name__ == "__main__":
    main()