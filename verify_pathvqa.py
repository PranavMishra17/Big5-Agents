#!/usr/bin/env python3
"""
Verify PathVQA dataset structure and count yes/no questions.
"""

import sys
import random
from collections import Counter
from datasets import load_dataset

def verify_pathvqa_dataset(max_samples=5000):
    """Verify PathVQA dataset and count yes/no questions."""
    print("Loading PathVQA dataset...")
    
    try:
        ds = load_dataset("flaviagiammarino/path-vqa", streaming=True)
        split_name = list(ds.keys())[0]
        print(f"Dataset split: {split_name}")
        
        answer_counts = Counter()
        total_questions = 0
        valid_images = 0
        sample_questions = []
        
        for i, question in enumerate(ds[split_name]):
            if i >= max_samples:  # Limit to avoid long processing
                break
                
            total_questions += 1
            
            # Check answer
            answer = question.get('answer', '').lower().strip()
            answer_counts[answer] += 1
            
            # Check image availability
            img = question.get('image')
            if img is not None:
                valid_images += 1
            
            # Collect first 10 samples for inspection
            if len(sample_questions) < 10:
                sample_questions.append({
                    'question': question.get('question', '')[:100] + "...",
                    'answer': answer,
                    'has_image': img is not None
                })
            
            if total_questions % 1000 == 0:
                print(f"Processed {total_questions} questions...")
        
        print(f"\n=== PathVQA Dataset Analysis (first {total_questions} questions) ===")
        print(f"Total questions processed: {total_questions}")
        print(f"Questions with valid images: {valid_images}")
        print(f"\nAnswer distribution:")
        for answer, count in sorted(answer_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_questions) * 100
            print(f"  {answer}: {count} ({percentage:.1f}%)")
        
        # Calculate yes/no questions
        yes_no_count = answer_counts.get('yes', 0) + answer_counts.get('no', 0)
        yes_no_percentage = (yes_no_count / total_questions) * 100
        print(f"\nYes/No questions: {yes_no_count} ({yes_no_percentage:.1f}%)")
        
        print(f"\n=== Sample Questions ===")
        for i, sample in enumerate(sample_questions, 1):
            print(f"{i}. Q: {sample['question']}")
            print(f"   A: {sample['answer']} | Has Image: {sample['has_image']}")
        
        # Estimate available yes/no questions for 1000 sample request
        if yes_no_count > 0:
            estimated_available = min(yes_no_count, int(yes_no_count * (max_samples / total_questions)))
            print(f"\nEstimated yes/no questions available: ~{estimated_available}")
            
            if estimated_available < 1000:
                print(f"⚠️  WARNING: Only ~{estimated_available} yes/no questions available, less than requested 1000!")
            else:
                print(f"✅ Sufficient yes/no questions available for 1000 sample request")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    verify_pathvqa_dataset()