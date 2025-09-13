#!/usr/bin/env python3
"""
Main pipeline runner that executes all three stages in sequence:
1. OS Model total_main
2. Critic Model batch_assess_all
3. Critic Response total_main
"""

import asyncio
import json
import argparse

def run_full_pipeline(video_dir='videos'):
    """Run the complete pipeline in sequence"""
    
    def reset_embedding_files():
        """Reset embedding files between stages"""
        open('embed_queries.json', 'w').close()
        open('ret_embeddings.json', 'w').close()
        with open('embed_queries.json', 'w') as f:
            json.dump({}, f, indent=2)
        with open('ret_embeddings.json', 'w') as f:
            json.dump({}, f, indent=2)
        print("Reset embedding files")
    
    print("="*80)
    print("STAGE 1: Running OS Model")
    print("="*80)
    
    # Reset embedding files before os_model
    reset_embedding_files()
    
    # Import and run os_model's total_main
    from os_model import total_main as os_total_main
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(os_total_main(video_dir))
    finally:
        loop.close()

    from reformat import reformat_answers

    reformat_answers(video_dir)
    
    print("\n" + "="*80)
    print("STAGE 2: Running Critic Batch Assessment")
    print("="*80)
    
    # Import and run critic_model_os's batch_assess_all
    from critic_model_os import batch_assess_all
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(batch_assess_all(video_dir))
    finally:
        loop.close()
    
    print("\n" + "="*80)
    print("STAGE 3: Running Critic Response Re-evaluation")
    print("="*80)
    
    # Reset embedding files before critic_response
    reset_embedding_files()
    
    # Import and run critic_response's total_main
    from critic_response import total_main as critic_total_main
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(critic_total_main(video_dir))
    finally:
        loop.close()
        
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete video analysis pipeline"
    )
    parser.add_argument(
        'video_dir',
        nargs='?',
        default='videos',
        help='Directory containing videos to process (default: videos)'
    )
    args = parser.parse_args()
    
    # Run the pipeline (now synchronous)
    run_full_pipeline(args.video_dir)

if __name__ == "__main__":
    main()