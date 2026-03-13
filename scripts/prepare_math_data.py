#!/usr/bin/env python3
"""
Prepare GRPO math reasoning training data
Convert GSM8K dataset to parquet format required by verl
"""

import os
import json
import pandas as pd
import re
from datasets import load_dataset
from typing import List, Dict, Any

def extract_solution(solution_str: str) -> str:
    """Extract final answer from solution string"""
    solution = re.search(r"#### (-?[0-9\.,]+)", solution_str)
    if solution is None:
        return ""
    final_solution = solution.group(1).replace(",", "")
    return final_solution

def format_gsm8k_data(dataset_path: str, output_dir: str) -> None:
    """Format GSM8K dataset for verl training format"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading GSM8K dataset...")
    try:
        # Try loading from local path
        if os.path.exists(dataset_path):
            dataset = load_dataset(dataset_path)
        else:
            # Load from HuggingFace
            dataset = load_dataset("openai/gsm8k", "main")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to HuggingFace dataset...")
        dataset = load_dataset("openai/gsm8k", "main")
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    def process_samples(split_name: str) -> List[Dict[str, Any]]:
        """Process dataset samples"""
        samples = []
        split_data = dataset[split_name]
        
        for i, item in enumerate(split_data):
            # Build training sample
            sample = {
                "prompt": item["question"],
                "response": item["answer"],
                "reward": 1.0,  # All answers in GSM8K dataset are correct, so reward is 1
                "answer": extract_solution(item["answer"]),
                "question": item["question"],
                "full_answer": item["answer"]
            }
            samples.append(sample)
            
            if i < 3:  # Show first 3 samples
                print(f"\nSample {i+1}:")
                print(f"Question: {sample['question'][:100]}...")
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
        "name": "GSM8K",
        "description": "Grade School Math 8K dataset for GRPO training",
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

def create_simple_math_dataset(output_dir: str, num_samples: int = 100) -> None:
    """Create a simple math dataset for testing"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple math problem templates
    math_problems = [
        {
            "question": "What is 2 + 3?",
            "answer": "5",
            "response": "To solve 2 + 3, I add the two numbers together: 2 + 3 = 5. #### 5"
        },
        {
            "question": "What is 10 - 4?",
            "answer": "6", 
            "response": "To solve 10 - 4, I subtract 4 from 10: 10 - 4 = 6. #### 6"
        },
        {
            "question": "What is 3 × 4?",
            "answer": "12",
            "response": "To solve 3 × 4, I multiply 3 by 4: 3 × 4 = 12. #### 12"
        },
        {
            "question": "What is 15 ÷ 3?",
            "answer": "5",
            "response": "To solve 15 ÷ 3, I divide 15 by 3: 15 ÷ 3 = 5. #### 5"
        },
        {
            "question": "What is 2²?",
            "answer": "4",
            "response": "To solve 2², I calculate 2 raised to the power of 2: 2² = 2 × 2 = 4. #### 4"
        }
    ]
    
    # Generate more samples
    samples = []
    for i in range(num_samples):
        problem = math_problems[i % len(math_problems)]
        sample = {
            "prompt": problem["question"],
            "response": problem["response"],
            "reward": 1.0,
            "answer": problem["answer"],
            "question": problem["question"],
            "full_answer": problem["response"],
            "reward_model": {
                "ground_truth": problem["answer"]
            }
        }
        samples.append(sample)
    
    # Save as parquet
    df = pd.DataFrame(samples)
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    # 80% training, 20% testing
    split_idx = int(len(samples) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Simple math dataset created:")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Train file: {train_path}")
    print(f"  - Test file: {test_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare math data for GRPO training")
    parser.add_argument("--dataset_path", default="/home/ubuntu/datasets/math_datasets/gsm8k", 
                       help="Path to GSM8K dataset")
    parser.add_argument("--output_dir", default="/home/ubuntu/data/gsm8k", 
                       help="Output directory for processed data")
    parser.add_argument("--simple", action="store_true", 
                       help="Create a simple test dataset instead")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for simple dataset")
    
    args = parser.parse_args()
    
    if args.simple:
        print("Creating simple math dataset...")
        create_simple_math_dataset(args.output_dir, args.num_samples)
    else:
        print("Preparing GSM8K dataset...")
        format_gsm8k_data(args.dataset_path, args.output_dir)
