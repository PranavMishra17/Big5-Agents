#!/usr/bin/env python3
"""
Count exact number of yes/no questions in PathVQA dataset.
"""

from datasets import load_dataset

def count_yesno_questions():
    """Count total yes/no questions available."""
    print("Loading PathVQA dataset to count yes/no questions...")
    
    try:
        ds = load_dataset("flaviagiammarino/path-vqa", streaming=True)
        split_name = list(ds.keys())[0]
        
        yes_count = 0
        no_count = 0
        total_processed = 0
        
        for question in ds[split_name]:
            answer = question.get('answer', '').lower().strip()
            if answer == 'yes':
                yes_count += 1
            elif answer == 'no':
                no_count += 1
            
            total_processed += 1
            
            if total_processed % 5000 == 0:
                current_yesno = yes_count + no_count
                print(f"Processed {total_processed}, Yes/No found: {current_yesno}")
                
                if total_processed >= 30000:  # Reasonable limit
                    break
        
        total_yesno = yes_count + no_count
        print(f"\n=== Final Count ===")
        print(f"Total processed: {total_processed}")
        print(f"Yes answers: {yes_count}")
        print(f"No answers: {no_count}")
        print(f"Total Yes/No: {total_yesno}")
        print(f"Yes/No percentage: {(total_yesno/total_processed)*100:.1f}%")
        
        if total_yesno >= 1000:
            print(f"✅ Sufficient yes/no questions available: {total_yesno}")
        else:
            print(f"⚠️ Limited yes/no questions: {total_yesno}")
            print(f"Recommend reducing --num-questions to {total_yesno}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    count_yesno_questions()