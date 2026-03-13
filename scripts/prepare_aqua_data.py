#!/usr/bin/env python3
"""
Prepare AQUA dataset for GRPO math reasoning training
Convert AQUA dataset to parquet format required by verl
"""

import os
import json
import pandas as pd
import re
from datasets import load_dataset
from typing import List, Dict, Any

def extract_solution(solution_str: str) -> str:
    """Extract final answer from solution string"""
    # AQUA format: answer is usually at the end or marked with specific patterns
    # Look for patterns like "The answer is X" or just extract the last word/number
    solution = re.search(r"(?:The answer is|Answer:|Final answer:)\s*([A-E])", solution_str, re.IGNORECASE)
    if solution is None:
        # Try to find the last letter A-E in the text
        solution = re.search(r"\b([A-E])\b", solution_str)
        if solution is None:
            return ""
    return solution.group(1)

def format_aqua_data(dataset_path: str, output_dir: str) -> None:
    """Format AQUA dataset for verl training format"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading AQUA dataset...")
    try:
        # Load from HuggingFace directly
        dataset = load_dataset("aqua_rat", "raw")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative AQUA dataset...")
        dataset = load_dataset("math_qa")
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    def process_samples(split_name: str) -> List[Dict[str, Any]]:
        """Process dataset samples"""
        samples = []
        split_data = dataset[split_name]
        
        for i, item in enumerate(split_data):
            # Build training sample
            question = item["question"]
            options = item["options"]
            correct = item["correct"]
            
            # Format options
            options_text = ""
            for j, option in enumerate(options):
                options_text += f"{chr(65+j)}. {option}\n"
            
            # Combine question and options
            full_question = f"{question}\n{options_text}"
            
            # Create response with reasoning
            response = f"I need to solve this step by step.\n{question}\n\nOptions:\n{options_text}\nAfter careful analysis, the correct answer is {correct}."
            
            sample = {
                "prompt": full_question,
                "response": response,
                "reward": 1.0,  # All answers in AQUA dataset are correct, so reward is 1
                "answer": correct,
                "question": question,
                "options": options,
                "full_answer": response
            }
            samples.append(sample)
            
            if i < 3:  # Show first 3 samples
                print(f"\nSample {i+1}:")
                print(f"Question: {sample['question'][:100]}...")
                print(f"Options: {sample['options']}")
                print(f"Answer: {sample['answer']}")
                print(f"Full response: {sample['response'][:200]}...")
        
        return samples
    
    # Process training set
    print("\nProcessing training data...")
    train_samples = process_samples("train")
    train_df = pd.DataFrame(train_samples)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"Training data saved to: {train_path}")
    print(f"Training samples: {len(train_samples)}")
    
    # Process test set
    print("\nProcessing test data...")
    test_samples = process_samples("test")
    test_df = pd.DataFrame(test_samples)
    test_path = os.path.join(output_dir, "test.parquet")
    test_df.to_parquet(test_path, index=False)
    print(f"Test data saved to: {test_path}")
    print(f"Test samples: {len(test_samples)}")
    
    # Create dataset info file
    dataset_info = {
        "name": "AQUA",
        "description": "AQUA dataset for GRPO training",
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "format": "parquet",
        "columns": list(train_df.columns)
    }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset info saved to: {info_path}")
    print("\nData preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {info_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare AQUA data for GRPO training")
    parser.add_argument("--dataset_path", default="/home/ubuntu/datasets/math_datasets/aqua", 
                       help="Path to AQUA dataset")
    parser.add_argument("--output_dir", default="/home/ubuntu/data/aqua", 
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    print("Preparing AQUA dataset...")
    format_aqua_data(args.dataset_path, args.output_dir)
