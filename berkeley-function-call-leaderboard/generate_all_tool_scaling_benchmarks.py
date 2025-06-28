#!/usr/bin/env python3
"""
Generate all tool scaling benchmarks with the comprehensive tool set.
"""

from bfcl_eval.scripts.tool_scaling_benchmark_efficient import ToolScalingBenchmarkEfficient
from pathlib import Path
import time

def main():
    # Initialize the generator
    generator = ToolScalingBenchmarkEfficient(
        data_dir=Path('bfcl_eval/data'),
        cache_dir=Path('bfcl_eval/data/embeddings_cache'),
        batch_size=200  # Optimized batch size for parallel processing
    )
    
    # Define all benchmark sizes
    benchmark_sizes = [5, 10, 20, 50, 128]
    
    print("🚀 Generating comprehensive tool scaling benchmarks with 33,090 tools")
    print("=" * 80)
    
    total_start_time = time.time()
    
    for num_tools in benchmark_sizes:
        print(f"\n📊 Generating {num_tools}-tool benchmark...")
        start_time = time.time()
        
        output_file = f'BFCL_v3_tool_scaling_{num_tools}.json'
        generator.generate_benchmark(num_tools=num_tools, output_file=output_file)
        
        elapsed = time.time() - start_time
        print(f"✅ Completed {num_tools}-tool benchmark in {elapsed:.1f} seconds")
        print(f"📁 Saved to: {output_file}")
    
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"🎉 All benchmarks generated successfully!")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds")
    print(f"🔧 Total tools available: 33,090")
    print(f"📊 Benchmarks created: {len(benchmark_sizes)}")
    
    # List generated files
    print("\n📁 Generated benchmark files:")
    for num_tools in benchmark_sizes:
        output_file = f'BFCL_v3_tool_scaling_{num_tools}.json'
        file_path = Path('bfcl_eval/data') / output_file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {output_file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {output_file} (not found)")

if __name__ == "__main__":
    main()