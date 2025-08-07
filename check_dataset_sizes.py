#!/usr/bin/env python3
"""
Dataset Size Checker
Check the total number of available questions in each dataset.
"""

import logging
import random
from typing import Dict, Any, List
from datasets import load_dataset
import pandas as pd
from pathlib import Path

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def check_medqa_size():
    """Check MedQA dataset size."""
    try:
        ds = load_dataset("sickgpt/001_MedQA_raw")
        questions = list(ds["train"])
        return len(questions)
    except Exception as e:
        logging.error(f"Error checking MedQA size: {str(e)}")
        return 0

def check_medmcqa_size():
    """Check MedMCQA dataset size."""
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        questions = list(ds["train"])
        return len(questions)
    except Exception as e:
        logging.error(f"Error checking MedMCQA size: {str(e)}")
        return 0

def check_pubmedqa_size():
    """Check PubMedQA dataset size."""
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        questions = list(ds["train"])
        return len(questions)
    except Exception as e:
        logging.error(f"Error checking PubMedQA size: {str(e)}")
        return 0

def check_mmlupro_med_size():
    """Check MMLU-Pro Medical dataset size."""
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        health_questions = []
        
        for split in ds.keys():
            for item in ds[split]:
                category = item.get("category", "").lower()
                if "health" in category:
                    health_questions.append(item)
        
        return len(health_questions)
    except Exception as e:
        logging.error(f"Error checking MMLU-Pro Medical size: {str(e)}")
        return 0

def check_ddxplus_size():
    """Check DDXPlus dataset size using the existing loading function."""
    try:
        # Import the existing loading function from dataset_runner
        from dataset_runner import load_ddxplus_dataset
        
        # Try to load a large number to see the actual available size
        # We'll use a reasonable number first, then estimate
        test_questions = load_ddxplus_dataset(num_questions=10000, random_seed=42)
        
        if test_questions:
            # If we got 10000, there might be more
            # Let's try a larger number
            more_questions = load_ddxplus_dataset(num_questions=50000, random_seed=42)
            if len(more_questions) == 50000:
                # There are at least 50k questions
                return 50000
            else:
                # Return the actual number we got
                return len(more_questions)
        else:
            return 0
            
    except Exception as e:
        logging.error(f"Error checking DDXPlus size: {str(e)}")
        return 0

def check_medbullets_size():
    """Check MedBullets dataset size."""
    try:
        ds = load_dataset("JesseLiu/medbulltes5op")
        
        # Check all splits
        total_questions = 0
        for split in ds.keys():
            questions = list(ds[split])
            total_questions += len(questions)
            logging.info(f"MedBullets {split}: {len(questions)} questions")
        
        return total_questions
    except Exception as e:
        logging.error(f"Error checking MedBullets size: {str(e)}")
        return 0

def check_pmc_vqa_size():
    """Check PMC-VQA dataset size efficiently without loading all images."""
    try:
        # Load dataset info without streaming to get size quickly
        ds = load_dataset("hamzamooraj99/PMC-VQA-1")
        
        total_questions = 0
        for split in ds.keys():
            count = len(ds[split])
            total_questions += count
            logging.info(f"PMC-VQA {split}: {count} questions")
        
        return total_questions
    except Exception as e:
        logging.error(f"Error checking PMC-VQA size: {str(e)}")
        return 0

def check_path_vqa_size():
    """Check Path-VQA dataset size efficiently without loading all images."""
    try:
        # Load dataset info without streaming to get size quickly
        ds = load_dataset("flaviagiammarino/path-vqa")
        
        total_questions = 0
        for split in ds.keys():
            count = len(ds[split])
            total_questions += count
            logging.info(f"Path-VQA {split}: {count} questions")
        
        return total_questions
    except Exception as e:
        logging.error(f"Error checking Path-VQA size: {str(e)}")
        return 0

def main():
    """Main function to check all dataset sizes."""
    setup_logging()
    
    print("🔍 Checking Dataset Sizes")
    print("=" * 50)
    
    datasets = {
        "MedQA": check_medqa_size,
        "MedMCQA": check_medmcqa_size,
        "PubMedQA": check_pubmedqa_size,
        "MMLU-Pro Medical": check_mmlupro_med_size,
        "DDXPlus": check_ddxplus_size,
        "MedBullets": check_medbullets_size,
        "PMC-VQA": check_pmc_vqa_size,
        "Path-VQA": check_path_vqa_size,
    }
    
    results = {}
    
    for dataset_name, check_func in datasets.items():
        print(f"\nChecking {dataset_name}...")
        try:
            size = check_func()
            results[dataset_name] = size
            print(f"✅ {dataset_name}: {size:,} questions")
        except Exception as e:
            print(f"❌ {dataset_name}: Error - {str(e)}")
            results[dataset_name] = 0
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DATASET SIZE SUMMARY")
    print("=" * 50)
    
    total_questions = 0
    for dataset_name, size in results.items():
        print(f"{dataset_name:20} | {size:>10,} questions")
        total_questions += size
    
    print("-" * 50)
    print(f"{'TOTAL':20} | {total_questions:>10,} questions")
    
    # Recommendations
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS")
    print("=" * 50)
    
    for dataset_name, size in results.items():
        if size >= 50000:
            print(f"✅ {dataset_name}: Excellent size ({size:,} questions) - can run 1000+ questions")
        elif size >= 20000:
            print(f"🟡 {dataset_name}: Good size ({size:,} questions) - can run 1000 questions")
        elif size >= 10000:
            print(f"🟠 {dataset_name}: Moderate size ({size:,} questions) - can run 500-1000 questions")
        elif size >= 5000:
            print(f"🔴 {dataset_name}: Small size ({size:,} questions) - limit to 500 questions")
        else:
            print(f"❌ {dataset_name}: Very small ({size:,} questions) - not suitable for large evaluation")

if __name__ == "__main__":
    main() 