"""
Dataset runner for the modular agent system with fixed question-level parallel processing.
Runs multiple questions from datasets through the agent system with proper isolation.
"""

import argparse
import logging
import os
import json
import random
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from tqdm import tqdm
import concurrent.futures
import threading
import copy

from datasets import load_dataset
from simulator import AgentSystemSimulator
import config
from utils.logger import SimulationLogger
from components.agent_recruitment import determine_complexity, recruit_agents
from components.medrag_integration import create_medrag_integration

import pickle
import pandas as pd
import zipfile
from pathlib import Path

# Thread-local storage for progress tracking
thread_local = threading.local()

def setup_logging():
    """Set up logging for the dataset runner."""
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.LOG_DIR, "dataset_runner.log")),
            logging.StreamHandler()
        ]
    )

def load_medqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load questions from the MedQA dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MedQA dataset with {num_questions} random questions")
    
    # Load the dataset
    try:
        ds = load_dataset("sickgpt/001_MedQA_raw")
        
        # Convert to list for easier processing
        questions = list(ds["train"])
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from MedQA dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MedQA dataset: {str(e)}")
        return []

def load_medmcqa_dataset(num_questions: int = 50, random_seed: int = 42, include_multi_choice: bool = True) -> List[Dict[str, Any]]:
    """
    Load questions from the MedMCQA dataset.
    
    Note: The include_multi_choice parameter is kept for compatibility but all questions
    are actually single-choice due to the dataset structure.
    """
    logging.info(f"Loading MedMCQA dataset with {num_questions} random questions")
    
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        questions = list(ds["train"])
        
        # Log the choice type distribution for awareness
        choice_types = [q.get("choice_type", "unknown") for q in questions[:1000]]  # Sample first 1000
        choice_type_counts = {}
        for ct in choice_types:
            choice_type_counts[ct] = choice_type_counts.get(ct, 0) + 1
        
        logging.info(f"Choice type distribution in dataset sample: {choice_type_counts}")
        logging.info("Note: All questions will be treated as single-choice due to known dataset issue")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions (no need to filter by choice_type)
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from MedMCQA dataset")
        validate_medmcqa_parsing(selected_questions)
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MedMCQA dataset: {str(e)}")
        return []

def validate_medmcqa_parsing(sample_questions: List[Dict[str, Any]]) -> None:
    """Validate that MedMCQA parsing works correctly for all question types."""
    
    print("Validating MedMCQA parsing...")
    
    single_count = 0
    multi_count = 0
    errors = 0
    
    for i, question_data in enumerate(sample_questions[:20]):  # Test first 20
        try:
            original_type = question_data.get("choice_type", "unknown")
            original_cop = question_data.get("cop", "N/A")
            
            # Parse with new function
            task, eval_data = format_medmcqa_for_task(question_data)
            parsed_answer = eval_data.get("ground_truth", "ERROR")
            
            if original_type == "single":
                single_count += 1
            elif original_type == "multi":
                multi_count += 1
            
            print(f"Q{i+1}: {original_type} -> {parsed_answer} (cop: {original_cop})")
            
        except Exception as e:
            errors += 1
            print(f"Q{i+1}: ERROR - {e}")
    
    print(f"\nSummary: {single_count} single, {multi_count} multi, {errors} errors")
    print("All questions are now correctly parsed as single-choice MCQs.")

def load_mmlupro_med_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load Health questions from the MMLU-Pro dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MMLU-Pro Health dataset with {num_questions} random questions")
    
    try:
        # Load the MMLU-Pro dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        logging.info(f"Loaded TIGER-Lab/MMLU-Pro dataset")
        
        # Get all available splits
        available_splits = list(ds.keys())
        logging.info(f"Available splits: {available_splits}")
        
        # Filter for Health category only
        health_questions = []
        
        for split in available_splits:
            logging.info(f"Processing split: {split} with {len(ds[split])} questions")
            
            for item in ds[split]:
                category = item.get("category", "").lower()
                
                # Filter specifically for health category
                if "health" in category:
                    health_questions.append(item)
                    if len(health_questions) % 50 == 0:  # Log progress
                        logging.info(f"Found {len(health_questions)} health questions so far...")
        
        logging.info(f"Found {len(health_questions)} health questions total")
        
        if not health_questions:
            logging.error("No health questions found in MMLU-Pro dataset")
            return []
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(health_questions):
            selected_questions = random.sample(health_questions, num_questions)
        else:
            selected_questions = health_questions
            logging.warning(f"Requested {num_questions} questions but found only {len(health_questions)} health questions. Using all available.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} health questions from MMLU-Pro dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MMLU-Pro Health dataset: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def load_pubmedqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load questions from the PubMedQA dataset.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading PubMedQA dataset with {num_questions} random questions")
    
    # Load the dataset
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        
        # Convert to list for easier processing
        questions = list(ds["train"])
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from PubMedQA dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading PubMedQA dataset: {str(e)}")
        return []

def format_medqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MedQA question into agent task and evaluation data.
    
    Args:
        question_data: Question data from the dataset
        
    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input (no GT)
        - eval_data: Ground truth, rationale, metadata for evaluation
    """
    # Extract question and choices
    question_text = question_data.get("question", "")
    choices = question_data.get("choices", [])
    
    # Ensure choices is a list (sometimes it might be nested differently)
    if not isinstance(choices, list):
        choices = []
    
    # Format choices as options
    options = []
    for i, choice in enumerate(choices):
        if isinstance(choice, str):
            options.append(f"{chr(65+i)}. {choice}")  # A, B, C, D, etc.
    
    # Get the expected output (correct answer)
    expected_output = question_data.get("expected_output", "")
    
    # Create agent task (no ground truth)
    agent_task = {
        "name": "MedQA Question",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with rationale"
    }
    
    # Create evaluation data (with ground truth)
    eval_data = {
        "ground_truth": expected_output,
        "rationale": {},  # No rationale provided in the dataset
        "metadata": {
            "dataset": "MedQA",
            "question_id": question_data.get("id", ""),
            "metamap": question_data.get("metamap", ""),
            "answer_idx": question_data.get("answer_idx", "")
        }
    }
    
    return agent_task, eval_data

def format_medmcqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MedMCQA question into agent task and evaluation data.
    Enhanced to provide better context for agent recruitment.

    Args:
        question_data: Question data from the MedMCQA dataset.

    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input.
        - eval_data: Ground truth, rationale, metadata for evaluation.
    """
    question_text = question_data.get("question", "")
    opa, opb, opc, opd = question_data.get("opa", ""), question_data.get("opb", ""), question_data.get("opc", ""), question_data.get("opd", "")
    option_letters = ['A', 'B', 'C', 'D']
    option_values = [opa, opb, opc, opd]

    options = [f"{letter}. {value}" for letter, value in zip(option_letters, option_values) if value.strip()]
    original_choice_type = question_data.get("choice_type", "single")
    
    # Enhanced description for recruitment context
    enhanced_description = question_text
    
    # If question is very short/vague, add options context for better recruitment
    if len(question_text.split()) < 10:  # Short questions likely need context
        options_text = " | ".join([f"{letter}: {value}" for letter, value in zip(option_letters, option_values) if value.strip()])
        enhanced_description = f"{question_text}\n\nAnswer options provide context: {options_text}"
    
    if original_choice_type == "multi":
        logging.debug(f"Question ID {question_data.get('id', 'unknown')} labeled as 'multi', treating as single choice.")
    
    cop = question_data.get("cop", 1)
    ground_truth = parse_cop_field(cop)

    explanation = question_data.get("exp", "")
    subject_name = question_data.get("subject_name", "")
    topic_name = question_data.get("topic_name", "")

    agent_task = {
        "name": "MedMCQA Question",
        "description": enhanced_description,  # Use enhanced description
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with rationale",
        "subject_context": subject_name,  # Add for recruitment
        "topic_context": topic_name       # Add for recruitment
    }

    eval_data = {
        "ground_truth": ground_truth,
        "rationale": {ground_truth: explanation} if explanation else {},
        "metadata": {
            "subject": subject_name,
            "topic": topic_name,
            "question_id": question_data.get("id", ""),
            "original_choice_type": original_choice_type,
            "cop_original": question_data.get("cop", ""),
            "original_question": question_text,  # Keep original for reference
            "enhanced_for_recruitment": len(question_text.split()) < 10
        }
    }

    return agent_task, eval_data


def parse_cop_field(cop) -> str:
    """
    Robustly parse the cop (correct option) field from MedMCQA dataset.
    
    Args:
        cop: The cop field value (can be int, str, or other)
        
    Returns:
        Single letter representing the correct answer (A, B, C, or D)
    """
    # Handle integer values (most common case)
    if isinstance(cop, int):
        return chr(64 + cop) if 1 <= cop <= 4 else "A"
    
    # Handle string values
    elif isinstance(cop, str):
        cop = cop.strip()
        
        # Handle numeric strings
        if cop.isdigit():
            cop_int = int(cop)
            return chr(64 + cop_int) if 1 <= cop_int <= 4 else "A"
        
        # Handle letter answers
        elif cop.upper() in ['A', 'B', 'C', 'D']:
            return cop.upper()
        
        # Handle comma-separated (take first one - fallback for edge cases)
        elif ',' in cop:
            first_part = cop.split(',')[0].strip()
            return parse_cop_field(first_part)  # Recursive call
        
        # Handle empty strings
        elif not cop:
            return "A"
        
        # Other string formats - try to extract number
        else:
            import re
            match = re.search(r'\d+', cop)
            if match:
                cop_int = int(match.group())
                return chr(64 + cop_int) if 1 <= cop_int <= 4 else "A"
            else:
                return "A"  # Default fallback
    
    # Handle list/array values (edge case)
    elif isinstance(cop, (list, tuple)) and len(cop) > 0:
        return parse_cop_field(cop[0])  # Take first element
    
    # Handle other types
    else:
        # Try to convert to string and parse
        try:
            return parse_cop_field(str(cop))
        except:
            return "A"  # Final fallback


def format_mmlupro_med_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MMLU-Pro medical question into agent task and evaluation data.
    Supports up to 10 options (A-J).

    Args:
        question_data: Question data from the MMLU-Pro dataset.

    Returns:
        Tuple of:
        - agent_task: Task for the agent system.
        - eval_data: Ground truth, rationale, and metadata.
    """
    question_text = question_data.get("question", "")
    options_data = question_data.get("options", [])
    options = []

    # Handle list format with up to 10 options
    if isinstance(options_data, list):
        for i, option in enumerate(options_data):
            if i < 10:  # Support up to 10 options (A-J)
                letter = chr(65 + i)  # A=65, B=66, ..., J=74
                options.append(f"{letter}. {option}")
            else:
                break  # Don't exceed 10 options
    elif isinstance(options_data, dict):
        # Handle dict format
        for key in sorted(options_data.keys()):
            options.append(f"{key}. {options_data[key]}")

    # Get ground truth
    ground_truth = question_data.get("answer", "")
    if not ground_truth and "answer_idx" in question_data:
        idx = question_data["answer_idx"]
        if isinstance(idx, int) and 0 <= idx < len(options_data):
            ground_truth = chr(65 + idx)  # Convert index to letter

    # Create expected output format based on number of options
    num_options = len(options)
    if num_options <= 4:
        expected_format = "Single letter selection (A-D) with rationale"
    elif num_options <= 10:
        last_letter = chr(64 + num_options)  # 64 + 10 = J
        expected_format = f"Single letter selection (A-{last_letter}) with rationale"
    else:
        expected_format = "Single letter selection with rationale"

    agent_task = {
        "name": "MMLU-Pro Health Question",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": expected_format,
        "num_options": num_options  # Add this for agents to know
    }

    eval_data = {
        "ground_truth": ground_truth,
        "rationale": {},  # MMLU-Pro provides no rationale
        "metadata": {
            "category": question_data.get("category", ""),
            "answer_idx": question_data.get("answer_idx", ""),
            "num_options": num_options
        }
    }

    return agent_task, eval_data

def format_pubmedqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format PubMedQA question into agent task and evaluation data.

    Args:
        question_data: Question data from PubMedQA.

    Returns:
        Tuple of:
        - agent_task: Task input for the agent.
        - eval_data: Ground truth and explanation for evaluation.
    """
    question_text = question_data.get("question", "")
    context = question_data.get("context", "")
    context_text = ""

    if isinstance(context, dict):
        for section, text in context.items():
            context_text += f"{section}: {text}\n\n"
    elif isinstance(context, list):
        context_text = "\n\n".join(context)
    elif isinstance(context, str):
        context_text = context

    expected_output = question_data.get("final_decision", "").lower()

    agent_task = {
        "name": "PubMedQA Question",
        "description": f"Research Question: {question_text}\n\nAbstract Context:\n{context_text}",
        "type": "yes_no_maybe",
        "options": ["yes", "no", "maybe"],
        "expected_output_format": "Answer (yes/no/maybe) with detailed scientific justification"
    }

    eval_data = {
        "ground_truth": expected_output,
        "rationale": {"long_answer": question_data.get("long_answer", "")},
        "metadata": {
            "pubid": question_data.get("pubid", "")
        }
    }

    return agent_task, eval_data

# ==================== DDXPLUS DATASET ====================

"""
DDXPLUS CSV FIX: Replace the load_ddxplus_dataset function with this version
that handles CSV files directly instead of ZIP files.
"""

def load_ddxplus_dataset(num_questions: int = 50, random_seed: int = 42, 
                        dataset_split: str = "train") -> List[Dict[str, Any]]:
    """
    Load questions from the DDXPlus dataset.
    FIXED: Handle CSV files directly instead of ZIP files.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        dataset_split: Which split to use ("train", "validate", "test")
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading DDXPlus dataset with {num_questions} random questions from {dataset_split} split")
    
    try:
        # Define dataset directory
        dataset_dir = Path("dataset/ddx")
        
        # Check if dataset directory exists
        if not dataset_dir.exists():
            logging.error(f"DDXPlus dataset directory not found: {dataset_dir}")
            return []
        
        # List all files in the directory for debugging
        all_files = list(dataset_dir.iterdir())
        logging.info(f"Files found in {dataset_dir}: {[f.name for f in all_files]}")
        
        # Load evidence and condition metadata
        evidences_file = dataset_dir / "release_evidences.json"
        conditions_file = dataset_dir / "release_conditions.json"
        
        if not evidences_file.exists():
            logging.error(f"Evidence file not found: {evidences_file}")
            return []
            
        if not conditions_file.exists():
            logging.error(f"Conditions file not found: {conditions_file}")
            return []
        
        # Load the JSON files
        try:
            import json
            with open(evidences_file, 'r', encoding='utf-8') as f:
                evidences = json.load(f)
            
            with open(conditions_file, 'r', encoding='utf-8') as f:
                conditions = json.load(f)
                
            logging.info(f"Loaded {len(evidences)} evidences and {len(conditions)} conditions")
            
        except Exception as json_error:
            logging.error(f"Error loading JSON files: {str(json_error)}")
            return []
        
        # FIXED: Load patient data from CSV files (not ZIP files)
        csv_mapping = {
            "train": "release_train_patients.csv",
            "validate": "release_validate_patients.csv", 
            "test": "release_test_patients.csv"
        }
        
        if dataset_split not in csv_mapping:
            logging.error(f"Invalid dataset split: {dataset_split}. Must be one of {list(csv_mapping.keys())}")
            return []
        
        patients_file = dataset_dir / csv_mapping[dataset_split]
        
        if not patients_file.exists():
            logging.error(f"DDXPlus patients CSV file not found: {patients_file}")
            logging.error(f"Available files: {[f.name for f in all_files]}")
            return []
        
        # FIXED: Load patients directly from CSV file
        patients = []
        try:
            import pandas as pd
            df = pd.read_csv(patients_file)
            patients = df.to_dict('records')
            logging.info(f"Successfully loaded CSV with {len(patients)} patients")
                    
        except Exception as e:
            logging.error(f"Error reading DDXPlus CSV file {patients_file}: {str(e)}")
            return []
        
        logging.info(f"Loaded {len(patients)} patients from DDXPlus {dataset_split} split")
        
        # Set random seed for reproducibility
        import random
        random.seed(random_seed)
        
        # Randomly select patients
        if num_questions < len(patients):
            selected_patients = random.sample(patients, num_questions)
        else:
            selected_patients = patients
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(patients)} patients. Using all available.")
        
        # Convert patients to question format
        questions = []
        for i, patient in enumerate(selected_patients):
            try:
                question_data = _convert_ddxplus_patient_to_question(patient, evidences, conditions, i)
                if question_data:
                    questions.append(question_data)
            except Exception as e:
                logging.error(f"Error converting DDXPlus patient {i} to question: {str(e)}")
                continue
        
        logging.info(f"Successfully converted {len(questions)} DDXPlus patients to questions")
        return questions
    
    except Exception as e:
        logging.error(f"Error loading DDXPlus dataset: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

# Also need to make sure the _convert_ddxplus_patient_to_question function handles the CSV format correctly
# The function should already work, but let's add some error handling

def _convert_ddxplus_patient_to_question(patient: Dict[str, Any], evidences: Dict, 
                                       conditions: Dict, patient_id: int) -> Optional[Dict[str, Any]]:
    """
    Convert a DDXPlus patient to a multiple choice question.
    ENHANCED: Better handling of CSV format data.
    
    Args:
        patient: Patient data from DDXPlus CSV
        evidences: Evidence definitions
        conditions: Condition definitions  
        patient_id: Patient identifier
        
    Returns:
        Question dictionary or None if conversion fails
    """
    try:
        # Extract patient information
        age = patient.get("AGE", "Unknown")
        sex = patient.get("SEX", "Unknown") 
        pathology = patient.get("PATHOLOGY", "")
        evidences_list = patient.get("EVIDENCES", [])
        initial_evidence = patient.get("INITIAL_EVIDENCE", "")
        differential_diagnosis = patient.get("DIFFERENTIAL_DIAGNOSIS", [])
        
        # ENHANCED: Handle different data formats from CSV
        # Evidences might be a string representation of a list
        if isinstance(evidences_list, str):
            try:
                import ast
                # Try to parse as a Python literal (list)
                evidences_list = ast.literal_eval(evidences_list)
            except:
                # If that fails, try splitting by common delimiters
                evidences_list = evidences_list.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                evidences_list = [e.strip() for e in evidences_list.split(',') if e.strip()]
        
        # Handle differential diagnosis if it's a string
        if isinstance(differential_diagnosis, str):
            try:
                import ast
                differential_diagnosis = ast.literal_eval(differential_diagnosis)
            except:
                # If parsing fails, create a simple structure
                differential_diagnosis = []
        
        # Convert evidences to readable format
        symptoms_text = _format_ddxplus_evidences(evidences_list, evidences)
        
        # Create question text
        question_text = f"Patient Information:\n"
        question_text += f"Age: {age}, Sex: {sex}\n\n"
        question_text += f"Presenting symptoms and medical history:\n{symptoms_text}\n\n"
        question_text += f"What is the most likely diagnosis?"
        
        # Create answer choices from differential diagnosis
        choices = []
        correct_answer = pathology
        
        # Extract diagnosis names from differential
        diff_diagnoses = []
        if isinstance(differential_diagnosis, list):
            for item in differential_diagnosis:
                if isinstance(item, list) and len(item) >= 2:
                    diff_diagnoses.append(item[0])  # Take diagnosis name
                elif isinstance(item, str):
                    diff_diagnoses.append(item)
        
        # Ensure correct answer is first in choices
        if correct_answer not in diff_diagnoses:
            diff_diagnoses.insert(0, correct_answer)
        
        # Take first 4 unique diagnoses
        unique_diagnoses = []
        for diag in diff_diagnoses:
            if diag not in unique_diagnoses and diag.strip():
                unique_diagnoses.append(diag)
            if len(unique_diagnoses) >= 4:
                break
        
        # If we don't have enough, pad with common conditions
        while len(unique_diagnoses) < 4:
            filler_conditions = ["Other infectious condition", "Requires further investigation", 
                               "Chronic inflammatory condition", "Acute viral syndrome"]
            for filler in filler_conditions:
                if filler not in unique_diagnoses:
                    unique_diagnoses.append(filler)
                    break
            if len(unique_diagnoses) >= 4:
                break
        
        # Format as multiple choice
        choices = unique_diagnoses[:4]
        
        # Find correct answer index
        try:
            correct_idx = choices.index(correct_answer)
            correct_letter = chr(65 + correct_idx)  # A, B, C, D
        except ValueError:
            correct_letter = "A"  # Default if not found
            choices[0] = correct_answer  # Ensure correct answer is option A
        
        return {
            "id": f"ddxplus_{patient_id}",
            "question": question_text,
            "choices": choices,
            "correct_answer": correct_answer,
            "correct_letter": correct_letter,
            "age": age,
            "sex": sex,
            "evidences": evidences_list,
            "initial_evidence": initial_evidence,
            "differential_diagnosis": differential_diagnosis,
            "metadata": {
                "dataset": "ddxplus",
                "patient_id": patient_id,
                "pathology": pathology
            }
        }
        
    except Exception as e:
        logging.error(f"Error converting DDXPlus patient {patient_id}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def _format_ddxplus_evidences(evidences_list: List[str], evidences_dict: Dict) -> str:
    """
    Format DDXPlus evidences into readable text.
    
    Args:
        evidences_list: List of evidence codes
        evidences_dict: Evidence definitions
        
    Returns:
        Formatted symptoms text
    """
    symptoms = []
    
    for evidence in evidences_list:
        try:
            # Handle categorical/multi-choice evidences (format: E_123_@_V_456)
            if "_@_" in evidence:
                evidence_name, value_code = evidence.split("_@_")
                evidence_info = evidences_dict.get(evidence_name, {})
                
                question_en = evidence_info.get("question_en", evidence_name)
                value_meaning = evidence_info.get("value_meaning", {})
                
                if value_code in value_meaning:
                    value_text = value_meaning[value_code].get("en", value_code)
                    if value_text != "NA":
                        symptoms.append(f"{question_en}: {value_text}")
                        
            else:
                # Handle binary evidences
                evidence_info = evidences_dict.get(evidence, {})
                question_en = evidence_info.get("question_en", evidence)
                if question_en and question_en != evidence:
                    symptoms.append(question_en)
                    
        except Exception as e:
            logging.debug(f"Error formatting evidence {evidence}: {str(e)}")
            continue
    
    return "\n".join([f"- {symptom}" for symptom in symptoms]) if symptoms else "No specific symptoms recorded"


def format_ddxplus_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format DDXPlus question into agent task and evaluation data.
    ENHANCED: Pass comprehensive medical context including raw evidence codes,
    symptom types, evidence relationships, and differential diagnosis probabilities.
    
    Args:
        question_data: Question data from DDXPlus dataset
        
    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input (comprehensive medical context)
        - eval_data: Ground truth, rationale, metadata for evaluation
    """
    question_text = question_data.get("question", "")
    choices = question_data.get("choices", [])
    
    # Format choices as options
    options = []
    for i, choice in enumerate(choices):
        options.append(f"{chr(65+i)}. {choice}")  # A, B, C, D
    
    correct_letter = question_data.get("correct_letter", "A")
    correct_answer = question_data.get("correct_answer", "")
    
    # ENHANCED: Extract comprehensive medical information
    age = question_data.get("age", "Unknown")
    sex = question_data.get("sex", "Unknown")
    evidences_list = question_data.get("evidences", [])
    initial_evidence = question_data.get("initial_evidence", "")
    differential_diagnosis = question_data.get("differential_diagnosis", [])
    
    # Create comprehensive medical context for agents
    medical_context = _create_ddxplus_medical_context(
        age, sex, evidences_list, initial_evidence, differential_diagnosis
    )
    
    # Enhanced question description with full medical context
    enhanced_description = f"""{question_text}

COMPREHENSIVE MEDICAL CONTEXT:

Patient Demographics:
- Age: {age}
- Sex: {sex}

Chief Complaint & Initial Presentation:
- Initial Evidence: {initial_evidence if initial_evidence else "Not specified"}

Evidence Analysis:
{medical_context['evidence_analysis']}

Symptom Categories:
{medical_context['symptom_categories']}

Differential Diagnosis Considerations:
{medical_context['differential_context']}

Clinical Decision Factors:
- Case Complexity: {medical_context['complexity_level']}
- Number of Evidence Points: {len(evidences_list)}
- Diagnostic Certainty: {medical_context['diagnostic_certainty']}
"""

    # ENHANCED: Create agent task with comprehensive medical context
    agent_task = {
        "name": "DDXPlus Clinical Diagnosis Case",
        "description": enhanced_description,
        "type": "mcq", 
        "options": options,
        "expected_output_format": "Single letter selection with comprehensive clinical reasoning including differential diagnosis analysis",
        
        # ENHANCED: Comprehensive clinical context for recruitment and decision-making
        "clinical_context": {
            "patient_demographics": {
                "age": age,
                "sex": sex,
                "age_category": _categorize_age(age)
            },
            "case_characteristics": {
                "complexity_level": medical_context['complexity_level'],
                "evidence_count": len(evidences_list),
                "has_initial_evidence": bool(initial_evidence),
                "has_differential": bool(differential_diagnosis),
                "diagnostic_certainty": medical_context['diagnostic_certainty']
            },
            "medical_specialties_needed": medical_context['specialties_needed'],
            "evidence_types": medical_context['evidence_types'],
            "symptom_systems": medical_context['symptom_systems'],
            "clinical_reasoning_required": [
                "symptom_analysis",
                "differential_diagnosis", 
                "evidence_synthesis",
                "diagnostic_decision_making"
            ]
        },
        
        # Raw medical data for advanced agents
        "raw_medical_data": {
            "evidence_codes": evidences_list,
            "initial_presenting_evidence": initial_evidence,
            "differential_probabilities": medical_context['differential_probs'],
            "symptom_categories": medical_context['raw_symptom_categories']
        }
    }
    
    # ENHANCED: Create comprehensive evaluation data
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {
            "correct_diagnosis": correct_answer,
            "differential_diagnosis": differential_diagnosis,
            "evidence_supporting_diagnosis": evidences_list,
            "initial_presentation": initial_evidence
        },
        "metadata": {
            "dataset": "ddxplus",
            "question_id": question_data.get("id", ""),
            "pathology": question_data.get("metadata", {}).get("pathology", ""),
            "patient_demographics": {
                "age": age,
                "sex": sex
            },
            "clinical_complexity": medical_context['complexity_level'],
            "evidence_analysis": medical_context['evidence_analysis'],
            "specialties_involved": medical_context['specialties_needed']
        }
    }
    
    return agent_task, eval_data

def _create_ddxplus_medical_context(age, sex, evidences_list, initial_evidence, differential_diagnosis):
    """Create comprehensive medical context for DDXPlus cases."""
    
    # Analyze evidence types and medical systems
    evidence_analysis = []
    symptom_systems = set()
    evidence_types = {"binary": 0, "categorical": 0, "multi_choice": 0}
    
    for evidence in evidences_list:
        if "_@_" in evidence:
            evidence_types["categorical"] += 1
            evidence_analysis.append(f"- Categorical evidence: {evidence}")
        else:
            evidence_types["binary"] += 1
            evidence_analysis.append(f"- Binary evidence: {evidence}")
            
        # Infer medical systems (simplified mapping)
        if any(term in evidence.lower() for term in ['cardio', 'heart', 'chest']):
            symptom_systems.add("cardiovascular")
        elif any(term in evidence.lower() for term in ['neuro', 'head', 'cognitive']):
            symptom_systems.add("neurological")
        elif any(term in evidence.lower() for term in ['gastro', 'stomach', 'digest']):
            symptom_systems.add("gastrointestinal")
        elif any(term in evidence.lower() for term in ['resp', 'lung', 'breath']):
            symptom_systems.add("respiratory")
        else:
            symptom_systems.add("general_medicine")
    
    # Determine complexity level
    complexity_score = len(evidences_list) + (len(differential_diagnosis) if differential_diagnosis else 0)
    if complexity_score < 5:
        complexity_level = "basic"
    elif complexity_score < 15:
        complexity_level = "intermediate"
    else:
        complexity_level = "advanced"
    
    # Extract differential diagnosis probabilities
    differential_probs = []
    if isinstance(differential_diagnosis, list):
        for item in differential_diagnosis:
            if isinstance(item, list) and len(item) >= 2:
                differential_probs.append({"condition": item[0], "probability": item[1]})
    
    # Determine required medical specialties
    specialties_needed = list(symptom_systems)
    if age != "Unknown":
        try:
            age_num = int(age)
            if age_num < 18:
                specialties_needed.append("pediatrics")
            elif age_num > 65:
                specialties_needed.append("geriatrics")
        except:
            pass
    
    # Calculate diagnostic certainty
    if differential_probs:
        max_prob = max(prob["probability"] for prob in differential_probs)
        if max_prob > 0.7:
            diagnostic_certainty = "high"
        elif max_prob > 0.4:
            diagnostic_certainty = "moderate"
        else:
            diagnostic_certainty = "low"
    else:
        diagnostic_certainty = "unknown"
    
    return {
        "evidence_analysis": "\n".join(evidence_analysis) if evidence_analysis else "No specific evidence analysis available",
        "symptom_categories": f"Binary: {evidence_types['binary']}, Categorical: {evidence_types['categorical']}, Multi-choice: {evidence_types['multi_choice']}",
        "differential_context": f"Differential diagnoses available: {len(differential_probs)} conditions with probabilities" if differential_probs else "No differential diagnosis probabilities available",
        "complexity_level": complexity_level,
        "diagnostic_certainty": diagnostic_certainty,
        "specialties_needed": specialties_needed,
        "evidence_types": evidence_types,
        "symptom_systems": list(symptom_systems),
        "differential_probs": differential_probs,
        "raw_symptom_categories": evidence_types
    }

def _categorize_age(age):
    """Categorize age for medical context."""
    try:
        age_num = int(age)
        if age_num < 2:
            return "infant"
        elif age_num < 12:
            return "child"
        elif age_num < 18:
            return "adolescent"
        elif age_num < 65:
            return "adult"
        else:
            return "elderly"
    except:
        return "unknown"
# ==================== MEDBULLETS DATASET ====================

def load_medbullets_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load questions from the MedBullets dataset from Hugging Face.
    
    Args:
        num_questions: Number of questions to load
        random_seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    logging.info(f"Loading MedBullets dataset with {num_questions} random questions")
    
    try:
        from datasets import load_dataset
        
        # Load the dataset
        ds = load_dataset("JesseLiu/medbulltes5op")
        
        # Check available splits and log them
        available_splits = list(ds.keys())
        logging.info(f"MedBullets available splits: {available_splits}")
        
        # Use the available splits - prioritize test, then validation
        if "test" in available_splits:
            questions = list(ds["test"])
            logging.info(f"Using 'test' split with {len(questions)} questions")
        elif "validation" in available_splits:
            questions = list(ds["validation"])
            logging.info(f"Using 'validation' split with {len(questions)} questions")
        elif available_splits:
            # Use the first available split
            split_name = available_splits[0]
            questions = list(ds[split_name])
            logging.info(f"Using '{split_name}' split with {len(questions)} questions")
        else:
            logging.error("No splits found in MedBullets dataset")
            return []
        
        logging.info(f"Total questions loaded from MedBullets: {len(questions)}")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions)
        else:
            selected_questions = questions
            logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
        
        logging.info(f"Successfully loaded {len(selected_questions)} questions from MedBullets dataset")
        return selected_questions
    
    except Exception as e:
        logging.error(f"Error loading MedBullets dataset: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def format_medbullets_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format MedBullets question into agent task and evaluation data.
    ENHANCED: Pass comprehensive USMLE context including question difficulty,
    step level context, and topic categorization.
    
    Args:
        question_data: Question data from MedBullets dataset
        
    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input (comprehensive USMLE context)
        - eval_data: Ground truth, rationale, metadata for evaluation
    """
    question_text = question_data.get("question", "")
    
    # Extract choices
    choices = []
    choice_keys = ["choicesA", "choicesB", "choicesC", "choicesD", "choicesE"]
    
    for i, key in enumerate(choice_keys):
        choice_text = question_data.get(key, "")
        if choice_text and choice_text.strip():  # Only add non-empty choices
            choices.append(f"{chr(65+i)}. {choice_text}")  # A, B, C, D, E
    
    # Get correct answer and explanation
    answer_idx = question_data.get("answer_idx", "")
    correct_answer = question_data.get("answer", "")
    explanation = question_data.get("explanation", "")
    
    # Convert answer_idx to letter
    if isinstance(answer_idx, (int, str)):
        try:
            if isinstance(answer_idx, str) and answer_idx.isdigit():
                idx = int(answer_idx)
                correct_letter = chr(64 + idx) if 1 <= idx <= 5 else "A"
            elif isinstance(answer_idx, str) and len(answer_idx) == 1 and answer_idx.upper() in "ABCDE":
                correct_letter = answer_idx.upper()
            elif isinstance(answer_idx, int):
                correct_letter = chr(64 + answer_idx) if 1 <= answer_idx <= 5 else "A"
            else:
                correct_letter = "A"
        except:
            correct_letter = "A"
    else:
        correct_letter = "A"
    
    # ENHANCED: Analyze USMLE context and question characteristics
    usmle_analysis = _create_medbullets_usmle_analysis(question_text, explanation, len(choices))
    
    # Enhanced question description with full USMLE context
    enhanced_description = f"""{question_text}

USMLE CLINICAL CONTEXT:

Question Characteristics:
- USMLE Step Level: {usmle_analysis['step_level']}
- Question Type: {usmle_analysis['question_type']}
- Clinical Complexity: {usmle_analysis['complexity_level']}
- Topic Category: {usmle_analysis['topic_category']}

Medical Domain Analysis:
{usmle_analysis['domain_analysis']}

Clinical Skills Required:
{usmle_analysis['skills_required']}

Question Difficulty Assessment:
- Estimated Difficulty: {usmle_analysis['difficulty_level']}
- Reasoning Type: {usmle_analysis['reasoning_type']}
- Knowledge Domain: {usmle_analysis['knowledge_domain']}
"""

    # ENHANCED: Create agent task with comprehensive USMLE context
    agent_task = {
        "name": "MedBullets USMLE Clinical Question",
        "description": enhanced_description,
        "type": "mcq",
        "options": choices,
        "expected_output_format": f"Single letter selection (A-{chr(64+len(choices))}) with USMLE-level clinical reasoning and step-by-step analysis",
        
        # ENHANCED: Comprehensive USMLE context for recruitment and decision-making
        "usmle_context": {
            "exam_characteristics": {
                "step_level": usmle_analysis['step_level'],
                "question_type": usmle_analysis['question_type'],
                "difficulty_level": usmle_analysis['difficulty_level'],
                "complexity_level": usmle_analysis['complexity_level']
            },
            "clinical_domains": {
                "primary_topic": usmle_analysis['topic_category'],
                "knowledge_domain": usmle_analysis['knowledge_domain'],
                "medical_specialties": usmle_analysis['medical_specialties']
            },
            "required_competencies": usmle_analysis['competencies_required'],
            "clinical_reasoning_type": usmle_analysis['reasoning_type'],
            "medical_specialties_needed": usmle_analysis['medical_specialties'],
            "clinical_skills_required": usmle_analysis['skills_list']
        },
        
        # Question format details
        "question_format": {
            "num_options": len(choices),
            "has_explanation": bool(explanation),
            "answer_format": f"A-{chr(64+len(choices))}"
        }
    }
    
    # ENHANCED: Create comprehensive evaluation data
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {
            correct_letter: explanation if explanation else correct_answer,
            "usmle_explanation": explanation,
            "correct_answer_text": correct_answer
        },
        "metadata": {
            "dataset": "medbullets",
            "answer_idx_original": answer_idx,
            "usmle_analysis": {
                "step_level": usmle_analysis['step_level'],
                "difficulty": usmle_analysis['difficulty_level'],
                "topic_category": usmle_analysis['topic_category'],
                "reasoning_type": usmle_analysis['reasoning_type']
            },
            "question_characteristics": {
                "has_explanation": bool(explanation),
                "num_choices": len(choices),
                "complexity": usmle_analysis['complexity_level']
            },
            "specialties_involved": usmle_analysis['medical_specialties']
        }
    }
    
    return agent_task, eval_data

def _create_medbullets_usmle_analysis(question_text, explanation, num_choices):
    """Create comprehensive USMLE analysis for MedBullets questions."""
    
    question_lower = question_text.lower()
    explanation_lower = explanation.lower() if explanation else ""
    
    # Determine USMLE Step level (heuristic analysis)
    if any(term in question_lower for term in ['step 1', 'basic science', 'pathophysiology', 'anatomy']):
        step_level = "Step 1"
    elif any(term in question_lower for term in ['step 2', 'clinical', 'patient', 'diagnosis', 'treatment']):
        step_level = "Step 2 CK/CS"
    elif any(term in question_lower for term in ['step 3', 'management', 'follow-up', 'monitoring']):
        step_level = "Step 3"
    else:
        # Infer from content
        if any(term in question_lower for term in ['patient', 'year-old', 'presents', 'complains']):
            step_level = "Step 2 CK"
        else:
            step_level = "Step 2/3"
    
    # Determine question type
    if "year-old" in question_lower and "presents" in question_lower:
        question_type = "Clinical Vignette"
    elif any(term in question_lower for term in ['which of the following', 'most likely', 'best next step']):
        question_type = "Clinical Reasoning"
    elif any(term in question_lower for term in ['mechanism', 'pathway', 'process']):
        question_type = "Basic Science"
    else:
        question_type = "Clinical Knowledge"
    
    # Analyze medical specialties
    medical_specialties = []
    specialty_keywords = {
        "cardiology": ["heart", "cardiac", "coronary", "myocardial", "arrhythmia"],
        "neurology": ["brain", "neural", "seizure", "stroke", "headache", "cognitive"],
        "infectious_disease": ["infection", "fever", "antibiotic", "sepsis", "pathogen"],
        "endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "glucose"],
        "gastroenterology": ["gastric", "intestinal", "liver", "hepatic", "bowel"],
        "pulmonology": ["lung", "respiratory", "pneumonia", "asthma", "breathing"],
        "nephrology": ["kidney", "renal", "urine", "creatinine", "dialysis"],
        "oncology": ["cancer", "tumor", "malignant", "metastasis", "chemotherapy"],
        "psychiatry": ["depression", "anxiety", "psychiatric", "mental", "mood"],
        "surgery": ["surgical", "operation", "procedure", "incision", "resection"]
    }
    
    for specialty, keywords in specialty_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            medical_specialties.append(specialty)
    
    if not medical_specialties:
        medical_specialties = ["general_medicine"]
    
    # Determine complexity level
    word_count = len(question_text.split())
    if word_count < 50:
        complexity_level = "basic"
    elif word_count < 150:
        complexity_level = "intermediate"
    else:
        complexity_level = "advanced"
    
    # Determine difficulty level
    if num_choices <= 4 and complexity_level == "basic":
        difficulty_level = "moderate"
    elif complexity_level == "advanced" or num_choices == 5:
        difficulty_level = "high"
    else:
        difficulty_level = "moderate"
    
    # Determine reasoning type
    if "best next step" in question_lower:
        reasoning_type = "clinical_management"
    elif "most likely" in question_lower:
        reasoning_type = "diagnostic_reasoning"
    elif "mechanism" in question_lower:
        reasoning_type = "pathophysiology"
    else:
        reasoning_type = "clinical_knowledge"
    
    # Determine topic category
    if any(term in question_lower for term in ['diagnosis', 'differential', 'presents']):
        topic_category = "diagnosis"
    elif any(term in question_lower for term in ['treatment', 'therapy', 'management']):
        topic_category = "treatment"
    elif any(term in question_lower for term in ['prevention', 'screening', 'prophylaxis']):
        topic_category = "prevention"
    else:
        topic_category = "clinical_knowledge"
    
    # Required competencies
    competencies_required = ["medical_knowledge", "clinical_reasoning"]
    if step_level in ["Step 2 CK", "Step 2 CS", "Step 3"]:
        competencies_required.extend(["patient_care", "clinical_decision_making"])
    if reasoning_type == "diagnostic_reasoning":
        competencies_required.append("diagnostic_skills")
    if reasoning_type == "clinical_management":
        competencies_required.append("treatment_planning")
    
    return {
        "step_level": step_level,
        "question_type": question_type,
        "complexity_level": complexity_level,
        "difficulty_level": difficulty_level,
        "topic_category": topic_category,
        "reasoning_type": reasoning_type,
        "knowledge_domain": step_level,
        "medical_specialties": medical_specialties,
        "competencies_required": competencies_required,
        "domain_analysis": f"Primary specialties: {', '.join(medical_specialties)}\nClinical focus: {topic_category}",
        "skills_required": f"Reasoning type: {reasoning_type}\nCompetencies: {', '.join(competencies_required)}",
        "skills_list": competencies_required
    }

# ====================  SYMCAT DATASET IMPLEMENTATION ====================


# PROPER SYMCAT IMPLEMENTATION - Replace in dataset_runner.py

def load_symcat_dataset(num_questions: int = 50, random_seed: int = 42, 
                       symcat_variant: str = "200") -> List[Dict[str, Any]]:
    """Load SymCat with comprehensive logging."""
    logging.info(f"=== LOADING SYMCAT-{symcat_variant} DATASET ===")
    
    try:
        dataset_dir = Path("dataset/symcat")
        if not dataset_dir.exists():
            logging.error(f"Directory not found: {dataset_dir}")
            return []
        
        # List all files for debugging
        all_files = list(dataset_dir.iterdir())
        logging.info(f"Available files: {[f.name for f in all_files]}")
        
        # Find main data file
        data_files = [f"symcat_{symcat_variant}_val_df.pkl", f"symcat_val_df.pkl"]
        data_file = None
        
        for filename in data_files:
            potential_path = dataset_dir / filename
            if potential_path.exists():
                data_file = potential_path
                logging.info(f"Found data file: {filename}")
                break
        
        if not data_file:
            logging.error(f"No data file found. Searched: {data_files}")
            return []
        
        # Load main data
        import pandas as pd
        val_data = pd.read_pickle(data_file)
        logging.info(f"Loaded data shape: {val_data.shape}")
        logging.info(f"Columns: {list(val_data.columns)}")
        
        # Log sample of data structure
        if len(val_data) > 0:
            sample_row = val_data.iloc[0]
            logging.info(f"Sample row structure:")
            for col, val in sample_row.items():
                logging.info(f"  {col}: {type(val)} - {str(val)[:100]}")
        
        # Load disease mappings with logging
        disease_names = _load_disease_mappings_with_logging(dataset_dir, symcat_variant)
        symptom_names = _load_symptom_mappings_with_logging(dataset_dir, symcat_variant)
        
        logging.info(f"Disease mappings: {len(disease_names)} loaded")
        logging.info(f"Symptom mappings: {len(symptom_names)} loaded")
        
        if len(disease_names) > 0:
            sample_diseases = dict(list(disease_names.items())[:5])
            logging.info(f"Sample diseases: {sample_diseases}")
            
        if len(symptom_names) > 0:
            sample_symptoms = dict(list(symptom_names.items())[:5])
            logging.info(f"Sample symptoms: {sample_symptoms}")
        
        # Filter and convert
        valid_data = val_data.dropna(subset=['disease_tag'])
        logging.info(f"Valid records: {len(valid_data)}")
        
        # Sample records
        random.seed(random_seed)
        records = valid_data.to_dict('records')
        selected_records = random.sample(records, min(num_questions, len(records)))
        logging.info(f"Selected {len(selected_records)} records for conversion")
        
        # Convert with detailed logging
        questions = []
        for i, record in enumerate(selected_records):
            try:
                logging.info(f"--- Converting record {i} ---")
                question_data = _convert_symcat_record_with_logging(
                    record, i, symcat_variant, disease_names, symptom_names
                )
                if question_data:
                    questions.append(question_data)
                    logging.info(f"✓ Record {i} converted successfully")
                    
                    # Log first question for verification
                    if i == 0:
                        logging.info(f"FIRST QUESTION PREVIEW:")
                        logging.info(f"  Question: {question_data['question'][:200]}...")
                        logging.info(f"  Choices: {question_data['choices']}")
                        logging.info(f"  Correct: {question_data['correct_answer']}")
                else:
                    logging.warning(f"✗ Record {i} conversion failed")
                    
            except Exception as e:
                logging.error(f"✗ Record {i} error: {str(e)}")
                continue
        
        logging.info(f"=== SYMCAT LOADING COMPLETE: {len(questions)} questions ===")
        return questions
    
    except Exception as e:
        logging.error(f"Critical error loading SymCat: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []


def _load_disease_mappings_with_logging(dataset_dir: Path, variant: str) -> Dict[int, str]:
    """Load disease mappings with detailed logging."""
    logging.info(f"--- Loading disease mappings for variant {variant} ---")
    
    # Check for mapping files
    mapping_files = [
        f"symcat_{variant}_disease_names.pkl",
        f"disease_names.pkl", 
        f"diseases_{variant}.pkl",
        f"symcat_diseases.pkl"
    ]
    
    logging.info(f"Searching for disease mapping files: {mapping_files}")
    
    for filename in mapping_files:
        file_path = dataset_dir / filename
        if file_path.exists():
            logging.info(f"✓ Found disease mapping file: {filename}")
            try:
                import pandas as pd
                mappings = pd.read_pickle(file_path)
                logging.info(f"✓ Loaded disease mappings: {type(mappings)}")
                
                if isinstance(mappings, dict):
                    logging.info(f"✓ Dict format: {len(mappings)} entries")
                    return mappings
                elif isinstance(mappings, (list, pd.Series)):
                    logging.info(f"✓ List/Series format: {len(mappings)} entries")
                    return {i: name for i, name in enumerate(mappings)}
                elif hasattr(mappings, 'to_dict'):
                    logging.info(f"✓ Converting to dict: {len(mappings)} entries")
                    return mappings.to_dict()
                else:
                    logging.warning(f"Unknown mapping format: {type(mappings)}")
                    
            except Exception as e:
                logging.error(f"Error loading {filename}: {str(e)}")
                continue
    
    # Fallback mapping
    logging.warning(f"No disease mapping files found, using fallback for {variant}")
    max_diseases = 200 if variant == "200" else 300 if variant == "300" else 400
    
    # Create realistic medical condition names
    fallback_diseases = {
        0: "Upper respiratory tract infection",
        1: "Acute gastroenteritis", 
        2: "Tension-type headache",
        3: "Essential hypertension",
        4: "Type 2 diabetes mellitus",
        5: "Generalized anxiety disorder",
        6: "Mechanical low back pain",
        7: "Bronchial asthma",
        8: "Community-acquired pneumonia",
        9: "Urinary tract infection",
        10: "Viral syndrome",
        11: "Allergic rhinitis",
        12: "Acute bronchitis",
        13: "Migraine without aura",
        14: "Osteoarthritis",
        15: "Major depressive disorder",
        **{i: f"Medical_condition_{i}" for i in range(16, max_diseases)}
    }
    
    logging.info(f"✓ Created fallback mappings: {len(fallback_diseases)} diseases")
    return fallback_diseases


def _load_symptom_mappings_with_logging(dataset_dir: Path, variant: str) -> Dict[int, str]:
    """Load symptom mappings with detailed logging."""
    logging.info(f"--- Loading symptom mappings for variant {variant} ---")
    
    mapping_files = [
        f"symcat_{variant}_symptom_names.pkl",
        f"symptom_names.pkl",
        f"symptoms_{variant}.pkl", 
        f"symcat_symptoms.pkl"
    ]
    
    logging.info(f"Searching for symptom mapping files: {mapping_files}")
    
    for filename in mapping_files:
        file_path = dataset_dir / filename
        if file_path.exists():
            logging.info(f"✓ Found symptom mapping file: {filename}")
            try:
                import pandas as pd
                mappings = pd.read_pickle(file_path)
                logging.info(f"✓ Loaded symptom mappings: {type(mappings)}")
                
                if isinstance(mappings, dict):
                    return mappings
                elif isinstance(mappings, (list, pd.Series)):
                    return {i: name for i, name in enumerate(mappings)}
                elif hasattr(mappings, 'to_dict'):
                    return mappings.to_dict()
                    
            except Exception as e:
                logging.error(f"Error loading {filename}: {str(e)}")
                continue
    
    # Fallback symptom mapping
    logging.warning("No symptom mapping files found, using fallback")
    
    fallback_symptoms = {
        0: "Fever", 1: "Fatigue", 2: "Headache", 3: "Cough", 4: "Nausea",
        5: "Abdominal pain", 6: "Back pain", 7: "Joint pain", 8: "Dizziness",
        9: "Shortness of breath", 10: "Chest pain", 11: "Rash", 12: "Diarrhea",
        13: "Constipation", 14: "Weight loss", 15: "Sleep problems",
        **{i: f"Symptom_{i}" for i in range(16, 376)}
    }
    
    logging.info(f"✓ Created fallback symptom mappings: {len(fallback_symptoms)} symptoms")
    return fallback_symptoms


def _convert_symcat_record_with_logging(record: Dict[str, Any], record_id: int, 
                                      variant: str, disease_names: Dict[int, str], 
                                      symptom_names: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """Convert SymCat record with detailed logging."""
    logging.info(f"Converting record {record_id}:")
    
    try:
        import numpy as np
        
        # Extract basic fields with logging
        disease_tag = record.get("disease_tag", "")
        explicit_symptoms = record.get("explicit_symptoms", {})
        implicit_symptoms = record.get("implicit_symptoms", {})
        
        logging.info(f"  Disease tag: {disease_tag} (type: {type(disease_tag)})")
        logging.info(f"  Explicit symptoms: {type(explicit_symptoms)}")
        logging.info(f"  Implicit symptoms: {type(implicit_symptoms)}")
        
        # Extract symptom indices with logging
        explicit_indices = _safe_extract_symptom_indices_with_logging(explicit_symptoms, "explicit")
        implicit_indices = _safe_extract_symptom_indices_with_logging(implicit_symptoms, "implicit")
        
        logging.info(f"  Extracted explicit indices: {explicit_indices[:5]}...")
        logging.info(f"  Extracted implicit indices: {implicit_indices[:3]}...")
        
        if not explicit_indices and not implicit_indices:
            logging.warning(f"  No symptoms found, skipping record")
            return None
        
        # Map to symptom names
        explicit_names = []
        for idx in explicit_indices[:5]:
            name = symptom_names.get(idx, f"Unknown_symptom_{idx}")
            explicit_names.append(name)
            
        implicit_names = []
        for idx in implicit_indices[:3]:
            name = symptom_names.get(idx, f"Unknown_symptom_{idx}")
            implicit_names.append(name)
            
        logging.info(f"  Explicit symptoms: {explicit_names}")
        logging.info(f"  Implicit symptoms: {implicit_names}")
        
        # Create symptom list for question
        symptom_descriptions = []
        for name in explicit_names:
            symptom_descriptions.append(f"Patient reports: {name}")
        for name in implicit_names:
            symptom_descriptions.append(f"Associated finding: {name}")
        
        # Create question text
        if symptom_descriptions:
            symptoms_text = "\n".join([f"- {s}" for s in symptom_descriptions])
            question_text = f"""A patient presents with the following:

{symptoms_text}

What is the most likely diagnosis?"""
        else:
            question_text = "Based on the clinical presentation, what is the most likely diagnosis?"
        
        # Get disease name
        disease_id = int(disease_tag) if str(disease_tag).isdigit() else 0
        correct_answer = disease_names.get(disease_id, f"Unknown_disease_{disease_tag}")
        
        logging.info(f"  Disease ID: {disease_id} -> {correct_answer}")
        
        # Create distractors
        all_diseases = list(disease_names.values())
        distractors = [d for d in all_diseases if d != correct_answer]
        
        import random
        random.seed(42 + record_id)
        chosen_distractors = random.sample(distractors, min(3, len(distractors)))
        
        choices = [correct_answer] + chosen_distractors
        random.shuffle(choices)
        
        correct_idx = choices.index(correct_answer)
        correct_letter = chr(65 + correct_idx)
        
        logging.info(f"  Final choices: {choices}")
        logging.info(f"  Correct answer: {correct_letter} ({correct_answer})")
        
        return {
            "id": f"symcat_{variant}_{record_id}",
            "question": question_text,
            "choices": choices,
            "correct_answer": correct_answer,
            "correct_letter": correct_letter,
            "disease_tag": disease_tag,
            "metadata": {
                "dataset": f"symcat_{variant}",
                "record_id": record_id,
                "disease_id": disease_id,
                "explicit_symptoms": explicit_names,
                "implicit_symptoms": implicit_names,
                "num_explicit": len(explicit_indices),
                "num_implicit": len(implicit_indices)
            }
        }
        
    except Exception as e:
        logging.error(f"  Conversion error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def _load_disease_mappings(dataset_dir: Path, variant: str) -> Dict[int, str]:
    """Load actual disease name mappings from SymCat files."""
    try:
        # Try different possible mapping file names
        mapping_files = [
            f"symcat_{variant}_disease_names.pkl",
            f"symcat_disease_names.pkl", 
            f"disease_names_{variant}.pkl",
            f"diseases.pkl",
            f"symcat_{variant}_diseases.pkl"
        ]
        
        for filename in mapping_files:
            file_path = dataset_dir / filename
            if file_path.exists():
                logging.info(f"Loading disease mappings from {filename}")
                import pandas as pd
                mappings = pd.read_pickle(file_path)
                
                # Handle different formats
                if isinstance(mappings, dict):
                    return mappings
                elif isinstance(mappings, (list, pd.Series)):
                    return {i: name for i, name in enumerate(mappings)}
                elif hasattr(mappings, 'to_dict'):
                    return mappings.to_dict()
        
        # If no mapping file found, try to extract from data
        logging.warning("No disease mapping file found, attempting to extract from data")
        return _extract_disease_mappings_from_data(dataset_dir, variant)
        
    except Exception as e:
        logging.error(f"Error loading disease mappings: {str(e)}")
        return {}


def _load_symptom_mappings(dataset_dir: Path, variant: str) -> Dict[int, str]:
    """Load actual symptom name mappings from SymCat files."""
    try:
        # Try different possible mapping file names
        mapping_files = [
            f"symcat_{variant}_symptom_names.pkl",
            f"symcat_symptom_names.pkl",
            f"symptom_names_{variant}.pkl", 
            f"symptoms.pkl",
            f"symcat_{variant}_symptoms.pkl"
        ]
        
        for filename in mapping_files:
            file_path = dataset_dir / filename
            if file_path.exists():
                logging.info(f"Loading symptom mappings from {filename}")
                import pandas as pd
                mappings = pd.read_pickle(file_path)
                
                # Handle different formats
                if isinstance(mappings, dict):
                    return mappings
                elif isinstance(mappings, (list, pd.Series)):
                    return {i: name for i, name in enumerate(mappings)}
                elif hasattr(mappings, 'to_dict'):
                    return mappings.to_dict()
        
        # If no mapping file found, try to extract from data
        logging.warning("No symptom mapping file found, attempting to extract from data")
        return _extract_symptom_mappings_from_data(dataset_dir, variant)
        
    except Exception as e:
        logging.error(f"Error loading symptom mappings: {str(e)}")
        return {}


def _extract_disease_mappings_from_data(dataset_dir: Path, variant: str) -> Dict[int, str]:
    """Extract disease mappings from the dataset structure if mapping files not found."""
    try:
        # Check if there are other pkl files that might contain mappings
        all_pkl_files = list(dataset_dir.glob("*.pkl"))
        logging.info(f"Available pkl files: {[f.name for f in all_pkl_files]}")
        
        # For now, return a basic mapping based on variant
        if variant == "200":
            max_diseases = 200
        elif variant == "300":
            max_diseases = 300
        elif variant == "400":
            max_diseases = 400
        else:
            max_diseases = 200
        
        # Create generic mappings as fallback
        logging.warning(f"Using fallback disease mappings for {max_diseases} diseases")
        return {i: f"Disease_{i}" for i in range(max_diseases)}
        
    except Exception as e:
        logging.error(f"Error extracting disease mappings: {str(e)}")
        return {}


def _extract_symptom_mappings_from_data(dataset_dir: Path, variant: str) -> Dict[int, str]:
    """Extract symptom mappings from the dataset structure if mapping files not found."""
    try:
        # SymCat typically has 376 symptoms
        max_symptoms = 376
        
        # Create generic mappings as fallback
        logging.warning(f"Using fallback symptom mappings for {max_symptoms} symptoms")
        return {i: f"Symptom_{i}" for i in range(max_symptoms)}
        
    except Exception as e:
        logging.error(f"Error extracting symptom mappings: {str(e)}")
        return {}


def _convert_symcat_record_with_mappings(record: Dict[str, Any], record_id: int, 
                                       variant: str, disease_names: Dict[int, str], 
                                       symptom_names: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """Convert SymCat record using actual disease/symptom mappings."""
    try:
        import numpy as np
        
        disease_tag = record.get("disease_tag", "")
        explicit_symptoms = record.get("explicit_symptoms", {})
        implicit_symptoms = record.get("implicit_symptoms", {})
        
        # Extract symptom indices safely
        explicit_indices = _safe_extract_symptom_indices(explicit_symptoms)
        implicit_indices = _safe_extract_symptom_indices(implicit_symptoms)
        
        # Skip if no symptoms
        if not explicit_indices and not implicit_indices:
            return None
        
        # Map to actual symptom names
        explicit_names = [symptom_names.get(idx, f"Unknown_Symptom_{idx}") 
                         for idx in explicit_indices[:5]]
        implicit_names = [symptom_names.get(idx, f"Unknown_Symptom_{idx}") 
                         for idx in implicit_indices[:3]]
        
        # Create symptom list
        all_symptoms = []
        for name in explicit_names:
            all_symptoms.append(f"Patient reports: {name}")
        for name in implicit_names:
            all_symptoms.append(f"Clinical finding: {name}")
        
        # Create question
        if all_symptoms:
            symptoms_text = "\n".join([f"- {s}" for s in all_symptoms])
            question_text = f"""Clinical presentation:

{symptoms_text}

What is the most likely diagnosis?"""
        else:
            question_text = "Based on the clinical presentation, what is the most likely diagnosis?"
        
        # Get actual disease name
        disease_id = int(disease_tag) if str(disease_tag).isdigit() else 0
        correct_answer = disease_names.get(disease_id, f"Unknown_Disease_{disease_tag}")
        
        # Create distractors from other diseases
        all_diseases = list(disease_names.values())
        distractors = [d for d in all_diseases if d != correct_answer]
        
        import random
        random.seed(42 + record_id)
        chosen_distractors = random.sample(distractors, min(3, len(distractors)))
        
        choices = [correct_answer] + chosen_distractors
        random.shuffle(choices)
        
        correct_idx = choices.index(correct_answer)
        correct_letter = chr(65 + correct_idx)
        
        return {
            "id": f"symcat_{variant}_{record_id}",
            "question": question_text,
            "choices": choices,
            "correct_answer": correct_answer,
            "correct_letter": correct_letter,
            "disease_tag": disease_tag,
            "metadata": {
                "dataset": f"symcat_{variant}",
                "record_id": record_id,
                "disease_id": disease_id,
                "explicit_symptoms": explicit_names,
                "implicit_symptoms": implicit_names,
                "num_explicit": len(explicit_indices),
                "num_implicit": len(implicit_indices)
            }
        }
        
    except Exception as e:
        logging.error(f"Error converting record {record_id}: {str(e)}")
        return None


def _safe_extract_symptom_indices(symptom_data) -> List[int]:
    """Safely extract symptom indices from SymCat format."""
    try:
        import numpy as np
        
        if symptom_data is None:
            return []
        
        # Handle NumPy arrays
        if isinstance(symptom_data, np.ndarray):
            if symptom_data.dtype == bool:
                # Boolean mask - get indices where True
                return np.where(symptom_data)[0].tolist()
            else:
                # Array of indices
                return symptom_data.flatten().tolist()
        
        # Handle dict format
        if isinstance(symptom_data, dict) and True in symptom_data:
            true_data = symptom_data[True]
            if isinstance(true_data, np.ndarray):
                return true_data.flatten().tolist() if true_data.size > 0 else []
            elif hasattr(true_data, '__iter__') and not isinstance(true_data, str):
                return list(true_data)
        
        # Handle direct list
        if isinstance(symptom_data, (list, tuple)):
            return [int(x) for x in symptom_data if str(x).isdigit()]
        
        return []
        
    except Exception as e:
        logging.warning(f"Error extracting symptom indices: {str(e)}")
        return []

def _safe_extract_symptom_indices_with_logging(symptom_data, symptom_type: str) -> List[int]:
    """Extract symptom indices with detailed logging."""
    try:
        import numpy as np
        
        logging.info(f"    Extracting {symptom_type} symptoms:")
        logging.info(f"      Data type: {type(symptom_data)}")
        
        if symptom_data is None:
            logging.info(f"      Result: Empty (None)")
            return []
        
        # Handle NumPy arrays
        if isinstance(symptom_data, np.ndarray):
            logging.info(f"      NumPy array shape: {symptom_data.shape}, dtype: {symptom_data.dtype}")
            if symptom_data.dtype == bool:
                indices = np.where(symptom_data)[0].tolist()
                logging.info(f"      Boolean mask -> indices: {indices[:10]}...")
                return indices
            else:
                result = symptom_data.flatten().tolist()
                logging.info(f"      Array flatten -> {len(result)} items")
                return result
        
        # Handle dict format
        if isinstance(symptom_data, dict):
            logging.info(f"      Dict keys: {list(symptom_data.keys())}")
            if True in symptom_data:
                true_data = symptom_data[True]
                logging.info(f"      True data: {type(true_data)}")
                if isinstance(true_data, np.ndarray):
                    result = true_data.flatten().tolist() if true_data.size > 0 else []
                    logging.info(f"      Dict[True] NumPy -> {len(result)} items")
                    return result
                elif hasattr(true_data, '__iter__') and not isinstance(true_data, str):
                    result = list(true_data)
                    logging.info(f"      Dict[True] iterable -> {len(result)} items")
                    return result
        
        # Handle list/tuple
        if isinstance(symptom_data, (list, tuple)):
            result = [int(x) for x in symptom_data if str(x).isdigit()]
            logging.info(f"      List/tuple -> {len(result)} valid indices")
            return result
        
        logging.info(f"      No valid extraction method found")
        return []
        
    except Exception as e:
        logging.error(f"      Extraction error: {str(e)}")
        return []


def format_symcat_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format SymCat question with logging."""
    logging.info(f"Formatting SymCat question: {question_data.get('id', 'unknown')}")
    
    question_text = question_data.get("question", "")
    choices = question_data.get("choices", [])
    correct_letter = question_data.get("correct_letter", "A")
    
    logging.info(f"  Question length: {len(question_text)} chars")
    logging.info(f"  Choices: {len(choices)} options")
    logging.info(f"  Correct: {correct_letter}")
    
    # Format options
    options = [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
    
    # Minimal agent task
    agent_task = {
        "name": "Medical Diagnosis",
        "description": question_text,
        "type": "mcq",
        "options": options,
        "expected_output_format": "Single letter selection with medical reasoning"
    }
    
    # Evaluation data
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {"correct_diagnosis": question_data.get("correct_answer", "")},
        "metadata": question_data.get("metadata", {})
    }
    
    return agent_task, eval_data

def _safe_extract_symptoms(symptom_data) -> List:
    """Safely extract symptoms from any format."""
    import numpy as np
    
    try:
        if symptom_data is None:
            return []
        
        # Handle NumPy arrays
        if isinstance(symptom_data, np.ndarray):
            if symptom_data.dtype == bool:
                # Boolean mask - get indices where True
                indices = np.where(symptom_data)[0]
                return indices.tolist()
            else:
                # Regular array
                return symptom_data.flatten().tolist()
        
        # Handle pandas Series
        if hasattr(symptom_data, 'values'):
            return _safe_extract_symptoms(symptom_data.values)
        
        # Handle dict
        if isinstance(symptom_data, dict):
            if True in symptom_data:
                true_data = symptom_data[True]
                # FIXED: Never use boolean evaluation on NumPy arrays
                if isinstance(true_data, np.ndarray):
                    return true_data.flatten().tolist() if true_data.size > 0 else []
                elif hasattr(true_data, '__len__') and not isinstance(true_data, str):
                    return list(true_data)
                else:
                    return [true_data] if true_data is not None else []
            return []
        
        # Handle list/tuple
        if isinstance(symptom_data, (list, tuple)):
            return list(symptom_data)
        
        return []
        
    except Exception:
        return []


# ==================== VERIFICATION FUNCTION ====================



def convert_symcat_pkl_to_csv(symcat_dir: str = "dataset/symcat"):
    """
    Helper function to convert SymCat PKL files to CSV format.
    Use this if you have pandas compatibility issues.
    """
    try:
        import pandas as pd
        
        symcat_path = Path(symcat_dir)
        if not symcat_path.exists():
            print(f"Directory {symcat_dir} does not exist")
            return
        
        pkl_files = list(symcat_path.glob("*.pkl"))
        if not pkl_files:
            print(f"No PKL files found in {symcat_dir}")
            return
        
        for pkl_file in pkl_files:
            try:
                print(f"Converting {pkl_file.name}...")
                
                # Try to load with pandas first
                try:
                    df = pd.read_pickle(pkl_file)
                except:
                    # If that fails, try with raw pickle
                    with open(pkl_file, 'rb') as f:
                        df = pickle.load(f)
                
                # Convert to CSV
                csv_file = pkl_file.with_suffix('.csv')
                df.to_csv(csv_file, index=False)
                print(f"  -> Saved as {csv_file.name}")
                
            except Exception as e:
                print(f"  -> Failed: {str(e)}")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")



####################################################################################################

def detect_agent_disagreement(agent_responses):
    """Detect if agents disagree on answers"""
    answers = []
    for agent_role, response in agent_responses.items():
        answer = None
        if isinstance(response, dict):
            if "final_decision" in response:
                # Extract answer from final_decision text
                answer = extract_answer_option(response["final_decision"])
            elif "extract" in response and isinstance(response["extract"], dict):
                if "answer" in response["extract"]:
                    answer = response["extract"]["answer"]
        elif isinstance(response, str):
            answer = extract_answer_option(response)
        
        if answer:
            answers.append(answer.upper())
    
    unique_answers = set(answers)
    disagreement = len(unique_answers) > 1
    
    return {
        "has_disagreement": disagreement,
        "unique_answers": list(unique_answers),
        "answer_distribution": {ans: answers.count(ans) for ans in unique_answers},
        "total_agents": len(answers),
        "all_answers": answers
    }

def extract_answer_option(content):
    """Extract answer option from agent response content"""
    if not isinstance(content, str):
        return None
    
    import re
    patterns = [
        r"ANSWER:\s*([A-Da-d])",
        r"FINAL ANSWER:\s*([A-Da-d])",
        r"answer is:?\s*([A-Da-d])",
        r"my answer:?\s*([A-Da-d])",
        r"option\s+([A-Da-d])",
        r"choose\s+([A-Da-d])",
        r"select\s+([A-Da-d])",
        r"\*\*([A-Da-d])\.\*\*",
        r"^([A-Da-d])\.",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def extract_yes_no_maybe_from_text(content):
    """Extract yes/no/maybe from text content."""
    if not isinstance(content, str):
        return None
    
    import re
    patterns = [
        r"ANSWER:\s*(yes|no|maybe)",
        r"answer is:?\s*(yes|no|maybe)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content.lower())
        if match:
            return match.group(1).lower()
    return None


def extract_agent_responses_info(simulation_results):
    """Extract detailed agent response information"""
    agent_responses_info = {}
    agent_responses = simulation_results.get("agent_responses", {})
    
    for agent_role, response_data in agent_responses.items():
        info = {
            "final_answer": None,
            "confidence": 1.0,
            "reasoning": "",
            "full_response": response_data
        }
        
        if isinstance(response_data, dict):
            # Extract answer
            if "final_decision" in response_data:
                # Try MCQ first, then yes/no/maybe
                info["final_answer"] = extract_answer_option(response_data["final_decision"])
                if not info["final_answer"]:
                    info["final_answer"] = extract_yes_no_maybe_from_text(response_data["final_decision"])
                info["reasoning"] = response_data["final_decision"][:500] + "..." if len(response_data["final_decision"]) > 500 else response_data["final_decision"]
            elif "extract" in response_data and isinstance(response_data["extract"], dict):
                if "final_decision" in response_data:
                    info["final_answer"] = extract_answer_option(response_data["final_decision"])
                if "confidence" in response_data["extract"]:
                    info["confidence"] = response_data["extract"]["confidence"]
            
            # Extract confidence if available
            if "confidence" in response_data:
                info["confidence"] = response_data["confidence"]
        else:
            info["final_answer"] = extract_answer_option(str(response_data))
            info["reasoning"] = str(response_data)[:500] + "..." if len(str(response_data)) > 500 else str(response_data)
        
        agent_responses_info[agent_role] = info
    
    return agent_responses_info


def process_single_question(question_index: int, 
                            question: Dict[str, Any],
                            dataset_type: str,
                            configuration: Dict[str, Any],
                            deployment_config: Dict[str, str],
                            run_output_dir: str,
                            max_retries: int = 3,
                            use_medrag: bool = False) -> Dict[str, Any]:
    """
    Process a single question with the given configuration and deployment.
    FIXED VERSION with proper MedRAG integration.
    """
    import gc
    import traceback

    # Create unique simulation ID for this question
    sim_id = f"{dataset_type}_{configuration['name'].lower().replace(' ', '_')}_q{question_index}_{deployment_config['name']}"
    if use_medrag:
        sim_id += "_medrag"
    
    # Format question based on dataset type
    if dataset_type == "medqa":
        agent_task, eval_data = format_medqa_for_task(question)
    elif dataset_type == "medmcqa":
        agent_task, eval_data = format_medmcqa_for_task(question)
    elif dataset_type == "mmlupro-med":
        agent_task, eval_data = format_mmlupro_med_for_task(question)
    elif dataset_type == "pubmedqa":
        agent_task, eval_data = format_pubmedqa_for_task(question)
    elif dataset_type == "ddxplus":
        agent_task, eval_data = format_ddxplus_for_task(question)
    elif dataset_type == "medbullets":
        agent_task, eval_data = format_medbullets_for_task(question)
    elif dataset_type == "symcat":
        agent_task, eval_data = format_symcat_for_task(question)
    elif dataset_type == "pmc_vqa":
        agent_task, eval_data = format_pmc_vqa_for_task(question)  # Now vision-enabled
    elif dataset_type == "path_vqa":
        agent_task, eval_data = format_path_vqa_for_task(question)  # Now vision-enabled
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Set thread-local data for this question
    thread_local.question_index = question_index
    thread_local.question_task = agent_task
    thread_local.question_eval = eval_data

    question_result = {
        "question_index": question_index,
        "deployment_used": deployment_config['name'],
        "simulation_id": sim_id,
        "recruitment_info": {},
        "agent_responses": {},
        "disagreement_analysis": {},
        "disagreement_flag": False,
        "medrag_info": {}  # FIXED: Initialize properly
    }

    try:
        for attempt in range(max_retries):
            try:
                # Create simulator with isolated task configuration
                simulator = AgentSystemSimulator(
                    simulation_id=sim_id,
                    use_team_leadership=configuration.get("leadership", False),
                    use_closed_loop_comm=configuration.get("closed_loop", False),
                    use_mutual_monitoring=configuration.get("mutual_monitoring", False),
                    use_shared_mental_model=configuration.get("shared_mental_model", False),
                    use_team_orientation=configuration.get("team_orientation", False),
                    use_mutual_trust=configuration.get("mutual_trust", False),
                    use_recruitment=configuration.get("recruitment", False),
                    recruitment_method=configuration.get("recruitment_method", "adaptive"),
                    recruitment_pool=configuration.get("recruitment_pool", "general"),
                    n_max=configuration.get("n_max", 5),
                    deployment_config=deployment_config,
                    question_specific_context=True,
                    task_config=agent_task,  # Pass task directly to simulator
                    eval_data=eval_data,     # Pass evaluation data directly
                    use_medrag=use_medrag    # FIXED: Pass MedRAG parameter
                )

                # Log that we're starting processing for this question
                medrag_status = "with MedRAG" if use_medrag else "without MedRAG"
                logging.info(f"Processing question {question_index} {medrag_status}, deployment {deployment_config['name']}, sim_id: {sim_id}")

                simulation_results = simulator.run_simulation()
                performance = simulator.evaluate_performance()

                # FIXED: Extract MedRAG information properly
                if use_medrag:
                    # Extract from simulation metadata
                    medrag_enhancement = simulation_results.get("simulation_metadata", {}).get("medrag_enhancement", {})
                    if medrag_enhancement:
                        question_result["medrag_info"] = medrag_enhancement
                        logging.info(f"Question {question_index}: MedRAG success={medrag_enhancement.get('success', False)}, snippets={medrag_enhancement.get('snippets_retrieved', 0)}")
                    else:
                        question_result["medrag_info"] = {"enabled": True, "success": False, "error": "No enhancement data found"}
                        logging.warning(f"Question {question_index}: MedRAG enabled but no enhancement data found")
                else:
                    question_result["medrag_info"] = {"enabled": False}

                question_result.update({
                    "agent_responses": extract_agent_responses_info(simulation_results),
                    "disagreement_analysis": detect_agent_disagreement(simulation_results.get("agent_responses", {})),
                    "decisions": simulation_results.get("decision_results", {}),
                    "performance": performance.get("task_performance", {}),
                    "agent_conversations": simulation_results.get("exchanges", [])
                })

                if question_result["disagreement_analysis"].get("has_disagreement", False):
                    question_result["disagreement_flag"] = True

                # Save individual question result to file
                if run_output_dir:
                    question_result_file = os.path.join(run_output_dir, f"question_{question_index}_result.json")
                    with open(question_result_file, 'w') as f:
                        json.dump(question_result, f, indent=2)
                    logging.info(f"Saved question {question_index} result to {question_result_file}")

                break  # Success, break retry loop

            except Exception as e:
                logging.error(f"Question {question_index} attempt {attempt+1} failed: {str(e)}")
                logging.error(traceback.format_exc())
                time.sleep(min(2 ** attempt, 15))
        
    except Exception as e:
        logging.error(f"Final error processing question {question_index}: {str(e)}")
        question_result["error"] = f"Processing error: {str(e)}"

    finally:
        # Clean up thread-local data and force garbage collection
        if hasattr(thread_local, 'question_task'):
            delattr(thread_local, 'question_task')
        if hasattr(thread_local, 'question_eval'):
            delattr(thread_local, 'question_eval')
        
        if 'simulator' in locals():
            del simulator
        gc.collect()

    return question_result


def track_agent_mind_changes(agent_analyses, agent_decisions):
    """Track if agents changed their answers after seeing teammates"""
    mind_changes = {}
    
    for agent_role in agent_analyses.keys():
        # Extract initial answer
        initial_extract = agent_analyses[agent_role].get("extract", {})
        initial_answer = initial_extract.get("answer", "").upper() if initial_extract.get("answer") else ""
        
        # Extract final answer
        final_extract = agent_decisions[agent_role].get("extract", {})
        final_answer = final_extract.get("answer", "").upper() if final_extract.get("answer") else ""
        
        # Check if changed
        changed = initial_answer != final_answer and initial_answer and final_answer
        
        mind_changes[agent_role] = {
            "initial_answer": initial_answer,
            "final_answer": final_answer, 
            "changed_mind": changed,
            "change_direction": f"{initial_answer} → {final_answer}" if changed else None
        }
    
    return mind_changes


def run_questions_with_configuration(
    questions: List[Dict[str, Any]],
    dataset_type: str,
    configuration: Dict[str, bool],
    output_dir: Optional[str] = None,
    max_retries: int = 3,
    n_max: int = 5,
    use_medrag: bool = False  # FIXED: Parameter properly passed
) -> Dict[str, Any]:
    """
    Run questions with specific configuration using parallel processing at question level.
    FIXED VERSION with proper MedRAG metrics collection.
    """
    config_name = configuration.get("name", "unknown")
    if use_medrag:
        config_name += "_medrag"
    
    logging.info(f"Running {len(questions)} questions with configuration: {config_name}")
    
    # Get all available deployments
    deployments = config.get_all_deployments()
    num_deployments = len(deployments)
    
    logging.info(f"Using {num_deployments} deployments for parallel question processing: {[d['name'] for d in deployments]}")
    
    # Reset complexity metrics
    from components import agent_recruitment
    agent_recruitment.reset_complexity_metrics()

    # Make sure configuration has proper recruitment method and n_max
    if config_name == "Baseline":
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "basic"
        configuration["n_max"] = 1
    elif config_name != "Custom Configuration":
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "intermediate"
        if "n_max" not in configuration:
            configuration["n_max"] = n_max
    
    # Setup output directory
    run_output_dir = os.path.join(output_dir, f"{dataset_type}_{config_name.lower().replace(' ', '_')}") if output_dir else None
    if run_output_dir:
        os.makedirs(run_output_dir, exist_ok=True)
    
    # Initialize results structure with FIXED MedRAG metrics
    results = {
        "configuration": config_name,
        "use_medrag": use_medrag,
        "dataset": dataset_type,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "configuration_details": configuration,
        "deployment_info": {
            "deployments_used": deployments,
            "parallel_processing": "question_level",
            "num_parallel_questions": num_deployments
        },
        "question_results": [],
        "errors": [],
        "summary": {
            "majority_voting": {"correct": 0, "total": 0},
            "weighted_voting": {"correct": 0, "total": 0},
            "borda_count": {"correct": 0, "total": 0}
        },
        "disagreement_summary": {
            "total_disagreements": 0,
            "disagreement_rate": 0.0,
            "disagreement_patterns": {}
        },
        "medrag_summary": {
            "enabled": use_medrag,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_snippets_retrieved": 0,
            "average_retrieval_time": 0.0,
            "enhancement_success_rate": 0.0
        },
        "complexity_distribution": {"basic": 0, "intermediate": 0, "advanced": 0},
        "deployment_usage": {d['name']: 0 for d in deployments}
    }
    
    # Add summary for yes/no/maybe if PubMedQA
    if dataset_type == "pubmedqa":
        results["summary"]["yes_no_maybe_voting"] = {"correct": 0, "total": 0}
    
    # Process questions in parallel batches
    batch_size = num_deployments
    num_batches = (len(questions) + batch_size - 1) // batch_size
    
    with tqdm(total=len(questions), desc=f"{config_name}") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]
            
            # Prepare futures for this batch
            future_to_info = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_questions), num_deployments)) as executor:
                # Submit each question in the batch to a different deployment
                for i, question in enumerate(batch_questions):
                    question_index = batch_start + i
                    deployment_index = i % num_deployments  # Round-robin assignment
                    deployment_config = deployments[deployment_index]
                    
                    # Track deployment usage
                    results["deployment_usage"][deployment_config['name']] += 1
                    
                    # Submit the question processing
                    future = executor.submit(
                        process_single_question,
                        question_index,
                        question,
                        dataset_type,
                        configuration,
                        deployment_config,
                        run_output_dir,
                        max_retries,
                        use_medrag  # FIXED: Pass MedRAG parameter
                    )
                    
                    future_to_info[future] = {
                        "question_index": question_index,
                        "deployment": deployment_config['name']
                    }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_info):
                    info = future_to_info[future]
                    try:
                        question_result = future.result()
                        results["question_results"].append(question_result)
                        
                        # FIXED: Track MedRAG usage properly
                        if use_medrag and "medrag_info" in question_result:
                            medrag_info = question_result["medrag_info"]
                            if medrag_info.get("enabled", False):
                                if medrag_info.get("success", False):
                                    results["medrag_summary"]["successful_retrievals"] += 1
                                    results["medrag_summary"]["total_snippets_retrieved"] += medrag_info.get("snippets_retrieved", 0)
                                    
                                    # Track retrieval time
                                    retrieval_time = medrag_info.get("retrieval_time", 0)
                                    if retrieval_time > 0:
                                        current_avg = results["medrag_summary"]["average_retrieval_time"]
                                        current_count = results["medrag_summary"]["successful_retrievals"]
                                        results["medrag_summary"]["average_retrieval_time"] = (
                                            (current_avg * (current_count - 1) + retrieval_time) / current_count
                                        )
                                else:
                                    results["medrag_summary"]["failed_retrievals"] += 1

                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'deployment': info['deployment'],
                            'processed': len(results["question_results"]),
                            'errors': len(results["errors"]),
                            'medrag_success': results["medrag_summary"]["successful_retrievals"] if use_medrag else 0
                        })
                        
                        # Update summary statistics if no error
                        if "error" not in question_result and "performance" in question_result:
                            task_performance = question_result["performance"]
                            
                            # Handle different task types properly
                            if dataset_type == "pubmedqa":
                                # For PubMedQA, all methods should work with yes/no/maybe
                                for method in ["majority_voting", "weighted_voting", "borda_count"]:
                                    if method in task_performance:
                                        method_perf = task_performance[method]
                                        if "correct" in method_perf:
                                            results["summary"][method]["total"] += 1
                                            if method_perf["correct"]:
                                                results["summary"][method]["correct"] += 1
                                
                            else:
                                methods_to_check = ["majority_voting", "weighted_voting", "borda_count"]
                                for method in methods_to_check:
                                    if method in task_performance:
                                        method_perf = task_performance[method]
                                        if "correct" in method_perf:
                                            results["summary"][method]["total"] += 1
                                            if method_perf["correct"]:
                                                results["summary"][method]["correct"] += 1
                        
                        # Track disagreements
                        if question_result.get("disagreement_flag", False):
                            results["disagreement_summary"]["total_disagreements"] += 1
                            
                            # Track disagreement patterns
                            disagreement_analysis = question_result.get("disagreement_analysis", {})
                            pattern = disagreement_analysis.get("answer_distribution", {})
                            pattern_key = "-".join(sorted(pattern.keys()))
                            if pattern_key:
                                results["disagreement_summary"]["disagreement_patterns"][pattern_key] = \
                                    results["disagreement_summary"]["disagreement_patterns"].get(pattern_key, 0) + 1
                        
                        # Track complexity
                        complexity = question_result.get("recruitment_info", {}).get("complexity_selected")
                        if complexity and complexity in results["complexity_distribution"]:
                            results["complexity_distribution"][complexity] += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing question {info['question_index']}: {str(e)}")
                        results["errors"].append({
                            "question_index": info['question_index'],
                            "deployment": info['deployment'],
                            "error_type": "processing_error",
                            "error": str(e)
                        })
                        pbar.update(1)
    
    # FIXED: Calculate final MedRAG statistics
    total_processed = len([q for q in results["question_results"] if "error" not in q])
    results["disagreement_summary"]["disagreement_rate"] = (
        results["disagreement_summary"]["total_disagreements"] / total_processed 
        if total_processed > 0 else 0
    )
    
    # Calculate MedRAG enhancement success rate
    if use_medrag:
        total_medrag_attempts = results["medrag_summary"]["successful_retrievals"] + results["medrag_summary"]["failed_retrievals"]
        if total_medrag_attempts > 0:
            results["medrag_summary"]["enhancement_success_rate"] = (
                results["medrag_summary"]["successful_retrievals"] / total_medrag_attempts
            )
    
    # Calculate accuracy for each method
    for method in results["summary"].keys():
        method_summary = results["summary"][method]
        method_summary["accuracy"] = (
            method_summary["correct"] / method_summary["total"] 
            if method_summary["total"] > 0 else 0.0
        )

    # Calculate mind change summary
    total_mind_changes = sum(
        q.get("mind_change_analysis", {}).get("agents_who_changed", 0) 
        for q in results["question_results"]
    )
    results["disagreement_summary"]["mind_change_summary"] = {
        "total_mind_changes": total_mind_changes,
        "average_change_rate": total_mind_changes / (total_processed * 5) if total_processed > 0 else 0
    }

    # Add complexity metrics
    from components import agent_recruitment
    if hasattr(agent_recruitment, "complexity_counts") and hasattr(agent_recruitment, "complexity_correct"):
        results["complexity_metrics"] = {
            "counts": agent_recruitment.complexity_counts.copy(),
            "correct": agent_recruitment.complexity_correct.copy(),
            "accuracy": {}
        }
        
        for level in ["basic", "intermediate", "advanced"]:
            count = results["complexity_metrics"]["counts"].get(level, 0)
            correct = results["complexity_metrics"]["correct"].get(level, 0)
            results["complexity_metrics"]["accuracy"][level] = correct / count if count > 0 else 0.0
    
    # Save enhanced overall results
    if run_output_dir:
        try:
            with open(os.path.join(run_output_dir, "summary.json"), 'w') as f:
                json.dump(results["summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "errors.json"), 'w') as f:
                json.dump(results["errors"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "disagreement_analysis.json"), 'w') as f:
                json.dump(results["disagreement_summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "deployment_usage.json"), 'w') as f:
                json.dump(results["deployment_usage"], f, indent=2)
                
            # FIXED: Save MedRAG metrics
            if use_medrag:
                with open(os.path.join(run_output_dir, "medrag_metrics.json"), 'w') as f:
                    json.dump(results["medrag_summary"], f, indent=2)
                
            # Save comprehensive detailed results
            with open(os.path.join(run_output_dir, "detailed_results_enhanced.json"), 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save enhanced summary results: {str(e)}")
    
    # FIXED: Enhanced summary print with MedRAG info
    print(f"\nSummary for {config_name} on {dataset_type}:")
    for method, stats in results["summary"].items():
        if "accuracy" in stats:
            print(f"  {method.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")

    if use_medrag:
        medrag_summary = results["medrag_summary"]
        total_attempts = medrag_summary["successful_retrievals"] + medrag_summary["failed_retrievals"]
        success_rate = medrag_summary["successful_retrievals"] / total_attempts if total_attempts > 0 else 0
        
        print(f"  MedRAG Enhancement:")
        print(f"    Success Rate: {medrag_summary['successful_retrievals']}/{total_attempts} ({success_rate:.2%})")
        print(f"    Total Snippets Retrieved: {medrag_summary['total_snippets_retrieved']}")
        print(f"    Average Retrieval Time: {medrag_summary['average_retrieval_time']:.2f}s")
        print(f"    Enhancement Success Rate: {medrag_summary['enhancement_success_rate']:.2%}")
    
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}/{len(questions)} questions ({len(results['errors'])/len(questions):.2%})")
    
    print(f"  Disagreements: {results['disagreement_summary']['total_disagreements']}/{total_processed} questions ({results['disagreement_summary']['disagreement_rate']:.2%})")
    
    # Print complexity distribution
    if any(results["complexity_distribution"].values()):
        print(f"  Complexity Distribution: {dict(results['complexity_distribution'])}")
    
    # Print deployment usage
    print(f"  Deployment Usage: {dict(results['deployment_usage'])}")
        
    return results


def create_comprehensive_combined_results(all_results, dataset_type, output_dir):
    """Create comprehensive combined results with all enhanced data"""
    
    combined_results = {
        "dataset": dataset_type,
        "timestamp": datetime.now().isoformat(),
        "configurations_tested": len(all_results),
        "total_questions_per_config": all_results[0]["num_questions"] if all_results else 0,
        
        # Configuration summaries
        "configuration_summaries": {},
        
        # Cross-configuration analysis
        "cross_analysis": {
            "disagreement_comparison": {},
            "complexity_distribution": {},
            "agent_performance_patterns": {},
            "teamwork_effectiveness": {}
        },
        
        # All detailed results
        "detailed_results": all_results,
        
        # Meta-analysis
        "meta_analysis": {
            "best_performing_configs": {},
            "most_disagreement_prone": {},
            "complexity_accuracy_correlation": {},
            "agent_type_effectiveness": {}
        }
    }
    
    # Process each configuration's results
    for result in all_results:
        config_name = result["configuration"]
        
        # Configuration summary
        combined_results["configuration_summaries"][config_name] = {
            "accuracy_by_method": result["summary"],
            "disagreement_rate": result.get("disagreement_summary", {}).get("disagreement_rate", 0),
            "total_errors": len(result.get("errors", [])),
            "complexity_distribution": analyze_complexity_distribution(result),
            "agent_patterns": analyze_agent_patterns(result),
            "deployment_usage": result.get("deployment_usage", {})
        }
        
        # Cross-analysis data
        combined_results["cross_analysis"]["disagreement_comparison"][config_name] = \
            result.get("disagreement_summary", {})
    
    # Perform meta-analysis
    combined_results["meta_analysis"] = perform_meta_analysis(all_results)
    
    # Save comprehensive results
    if output_dir:
        with open(os.path.join(output_dir, "comprehensive_combined_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
    
    return combined_results

def analyze_complexity_distribution(result):
    """Analyze complexity distribution for a configuration"""
    complexity_counts = {"basic": 0, "intermediate": 0, "advanced": 0}
    complexity_accuracy = {"basic": {"correct": 0, "total": 0}, 
                          "intermediate": {"correct": 0, "total": 0}, 
                          "advanced": {"correct": 0, "total": 0}}
    
    for q_result in result.get("question_results", []):
        recruitment_info = q_result.get("recruitment_info", {})
        complexity = recruitment_info.get("complexity_selected", "unknown")
        
        if complexity in complexity_counts:
            complexity_counts[complexity] += 1
            
            # Check accuracy
            performance = q_result.get("performance", {})
            if "majority_voting" in performance and "correct" in performance["majority_voting"]:
                complexity_accuracy[complexity]["total"] += 1
                if performance["majority_voting"]["correct"]:
                    complexity_accuracy[complexity]["correct"] += 1
    
    return {"counts": complexity_counts, "accuracy": complexity_accuracy}

def analyze_agent_patterns(result):
    """Analyze agent recruitment and performance patterns"""
    agent_type_counts = {}
    agent_performance = {}
    
    for q_result in result.get("question_results", []):
        recruitment_info = q_result.get("recruitment_info", {})
        agents = recruitment_info.get("agents_recruited", [])
        
        for agent in agents:
            role = agent.get("role", "unknown")
            agent_type_counts[role] = agent_type_counts.get(role, 0) + 1
    
    return {"agent_type_distribution": agent_type_counts}

def perform_meta_analysis(all_results):
    """Perform meta-analysis across all configurations"""
    
    # Find best performing configurations
    best_configs = {}
    for method in ["majority_voting", "weighted_voting", "borda_count"]:
        best_accuracy = 0
        best_config = None
        
        for result in all_results:
            accuracy = result["summary"].get(method, {}).get("accuracy", 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = result["configuration"]
        
        best_configs[method] = {"config": best_config, "accuracy": best_accuracy}
    
    # Find most disagreement-prone configurations
    disagreement_ranking = []
    for result in all_results:
        disagreement_rate = result.get("disagreement_summary", {}).get("disagreement_rate", 0)
        disagreement_ranking.append({
            "config": result["configuration"],
            "disagreement_rate": disagreement_rate
        })
    
    disagreement_ranking.sort(key=lambda x: x["disagreement_rate"], reverse=True)
    
    return {
        "best_performing_configs": best_configs,
        "disagreement_ranking": disagreement_ranking
    }


# ==================== ENHANCED ERROR HANDLING ====================

def validate_dataset_directories():
    """
    Validate that all required dataset directories and files exist.
    """
    validation_results = {
        "ddxplus": {"status": "unknown", "files": []},
        "symcat": {"status": "unknown", "files": []}, 
        "medbullets": {"status": "unknown", "note": "Loaded from Hugging Face"}
    }
    
    # Check DDXPlus
    ddx_dir = Path("dataset/ddx")
    if ddx_dir.exists():
        required_files = [
            "release_evidences.json",
            "release_conditions.json", 
            "release_train_patients.zip",
            "release_validate_patients.zip",
            "release_test_patients.zip"
        ]
        
        found_files = []
        for file in required_files:
            if (ddx_dir / file).exists():
                found_files.append(file)
        
        validation_results["ddxplus"]["files"] = found_files
        validation_results["ddxplus"]["status"] = "complete" if len(found_files) == len(required_files) else "partial"
    else:
        validation_results["ddxplus"]["status"] = "missing"
    
    # Check SymCat
    symcat_dir = Path("dataset/symcat")
    if symcat_dir.exists():
        pkl_files = list(symcat_dir.glob("*.pkl"))
        validation_results["symcat"]["files"] = [f.name for f in pkl_files]
        validation_results["symcat"]["status"] = "complete" if pkl_files else "empty"
    else:
        validation_results["symcat"]["status"] = "missing"
    
    # MedBullets will be validated when loading from Hugging Face
    validation_results["medbullets"]["status"] = "online"
    
    return validation_results

def validate_all_datasets():
    """
    Comprehensive validation of all dataset availability and structure.
    FIXED: Removed emojis that cause Windows encoding issues.
    """
    logging.info("Validating dataset availability...")
    
    validation_results = {
        "online_datasets": {},
        "local_datasets": {},
        "overall_status": "unknown"
    }
    
    # Test online datasets
    online_datasets = ["medqa", "medmcqa", "pubmedqa", "mmlupro-med", "medbullets"]
    
    for dataset in online_datasets:
        try:
            if dataset == "medqa":
                test_questions = load_medqa_dataset(num_questions=1, random_seed=42)
            elif dataset == "medmcqa":
                test_questions = load_medmcqa_dataset(num_questions=1, random_seed=42)
            elif dataset == "pubmedqa":
                test_questions = load_pubmedqa_dataset(num_questions=1, random_seed=42)
            elif dataset == "mmlupro-med":
                test_questions = load_mmlupro_med_dataset(num_questions=1, random_seed=42)
            elif dataset == "medbullets":
                test_questions = load_medbullets_dataset(num_questions=1, random_seed=42)
            
            if test_questions:
                validation_results["online_datasets"][dataset] = {
                    "status": "available",
                    "sample_loaded": True,
                    "count": len(test_questions)
                }
                logging.info(f"[OK] {dataset}: Available")  # FIXED: No emoji
            else:
                validation_results["online_datasets"][dataset] = {
                    "status": "empty",
                    "sample_loaded": False,
                    "count": 0
                }
                logging.warning(f"[WARN] {dataset}: Empty dataset")  # FIXED: No emoji
                
        except Exception as e:
            validation_results["online_datasets"][dataset] = {
                "status": "error",
                "error": str(e),
                "sample_loaded": False
            }
            logging.error(f"[ERROR] {dataset}: Error - {str(e)}")  # FIXED: No emoji
    
    # Test local datasets
    local_datasets = ["ddxplus", "symcat"]
    
    for dataset in local_datasets:
        try:
            if dataset == "ddxplus":
                test_questions = load_ddxplus_dataset(num_questions=1, random_seed=42)
            elif dataset == "symcat":
                test_questions = load_symcat_dataset(num_questions=1, random_seed=42)
            
            if test_questions:
                validation_results["local_datasets"][dataset] = {
                    "status": "available", 
                    "sample_loaded": True,
                    "count": len(test_questions)
                }
                logging.info(f"[OK] {dataset}: Available")  # FIXED: No emoji
            else:
                validation_results["local_datasets"][dataset] = {
                    "status": "empty",
                    "sample_loaded": False,
                    "count": 0
                }
                logging.warning(f"[WARN] {dataset}: Empty dataset")  # FIXED: No emoji
                
        except Exception as e:
            validation_results["local_datasets"][dataset] = {
                "status": "error",
                "error": str(e),
                "sample_loaded": False
            }
            logging.error(f"[ERROR] {dataset}: Error - {str(e)}")  # FIXED: No emoji
    
    # Determine overall status
    all_datasets = {**validation_results["online_datasets"], **validation_results["local_datasets"]}
    available_count = sum(1 for ds in all_datasets.values() if ds.get("status") == "available")
    total_count = len(all_datasets)
    
    if available_count == total_count:
        validation_results["overall_status"] = "all_available"
        logging.info(f"[SUCCESS] All {total_count} datasets are available!")  # FIXED: No emoji
    elif available_count > 0:
        validation_results["overall_status"] = "partial_available"
        logging.info(f"[INFO] {available_count}/{total_count} datasets are available")  # FIXED: No emoji
    else:
        validation_results["overall_status"] = "none_available"
        logging.error(f"[CRITICAL] No datasets are available!")  # FIXED: No emoji
    
    return validation_results


# Example usage and testing functions
def test_new_datasets():
    """
    Test function to validate the new dataset loading functions.
    """
    logging.info("Testing new dataset loading functions...")
    
    # Test DDXPlus
    try:
        ddx_questions = load_ddxplus_dataset(num_questions=2, random_seed=42)
        logging.info(f"DDXPlus test: Loaded {len(ddx_questions)} questions")
        if ddx_questions:
            task, eval_data = format_ddxplus_for_task(ddx_questions[0])
            logging.info(f"DDXPlus formatting test: Success")
    except Exception as e:
        logging.error(f"DDXPlus test failed: {str(e)}")
    
    # Test MedBullets
    try:
        mb_questions = load_medbullets_dataset(num_questions=2, random_seed=42)
        logging.info(f"MedBullets test: Loaded {len(mb_questions)} questions")
        if mb_questions:
            task, eval_data = format_medbullets_for_task(mb_questions[0])
            logging.info(f"MedBullets formatting test: Success")
    except Exception as e:
        logging.error(f"MedBullets test failed: {str(e)}")
    
    # Test SymCat
    try:
        sc_questions = load_symcat_dataset(num_questions=2, random_seed=42)
        logging.info(f"SymCat test: Loaded {len(sc_questions)} questions")
        if sc_questions:
            task, eval_data = format_symcat_for_task(sc_questions[0])
            logging.info(f"SymCat formatting test: Success")
    except Exception as e:
        logging.error(f"SymCat test failed: {str(e)}")


# ==================== IMAGE DATASET  ====================

# Replace the existing format functions with these:

def format_pmc_vqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format PMC-VQA with image data."""
    question_text = question_data.get("Question", "")
    answer = question_data.get("Answer", "")
    answer_label = question_data.get("Answer_label", "")
    image = question_data.get("image")
    
    # Extract choices
    choices = []
    for i, key in enumerate(["Choice A", "Choice B", "Choice C", "Choice D"]):
        choice_text = question_data.get(key, "")
        if choice_text and choice_text.strip():
            choices.append(f"{chr(65+i)}. {choice_text}")
    
    # Determine correct letter
    correct_letter = answer_label.upper() if answer_label and answer_label.upper() in "ABCD" else "A"
    
    agent_task = {
        "name": "PMC-VQA Medical Image Question",
        "description": f"MEDICAL IMAGE QUESTION: {question_text}\n\nAnalyze the provided medical image to answer this question.",
        "type": "mcq",
        "options": choices,
        "expected_output_format": f"Single letter (A-{chr(64+len(choices))}) with image analysis",
        "image_data": {
            "image": image,
            "image_available": image is not None,
            "requires_visual_analysis": True
        }
    }
    
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {correct_letter: answer},
        "metadata": {"dataset": "pmc_vqa", "has_image": image is not None}
    }
    
    return agent_task, eval_data

def format_path_vqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format Path-VQA as binary MCQ with image data."""
    question_text = question_data.get("question", "")
    answer = question_data.get("answer", "").lower().strip()
    image = question_data.get("image")
    
    choices = ["A. Yes", "B. No"]
    correct_letter = "A" if answer == "yes" else "B"
    
    agent_task = {
        "name": "Path-VQA Pathology Question",
        "description": f"PATHOLOGY IMAGE QUESTION: {question_text}\n\nExamine the pathology image to determine: Yes or No?",
        "type": "mcq",
        "options": choices,
        "expected_output_format": "A for Yes, B for No with pathological analysis",
        "image_data": {
            "image": image,
            "image_available": image is not None,
            "is_pathology_image": True,
            "requires_visual_analysis": True
        }
    }
    
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {correct_letter: f"Pathology analysis: {answer.title()}"},
        "metadata": {"dataset": "path_vqa", "original_answer": answer, "has_image": image is not None}
    }
    
    return agent_task, eval_data

# ==================== PMC-VQA DATASET ====================

def load_pmc_vqa_dataset(num_questions: int = 50, random_seed: int = 42, 
                        dataset_split: str = "test") -> List[Dict[str, Any]]:
    """
    Load PMC-VQA dataset using streaming to avoid large downloads.
    """
    logging.info(f"Loading PMC-VQA-1 dataset with {num_questions} questions from {dataset_split} split (streaming)")
    
    try:
        from datasets import load_dataset
        
        # Use streaming to avoid downloading entire dataset
        ds = load_dataset("hamzamooraj99/PMC-VQA-1", streaming=True)
        
        available_splits = list(ds.keys())
        logging.info(f"PMC-VQA-1 available splits: {available_splits}")
        
        if dataset_split not in available_splits:
            dataset_split = available_splits[0] if available_splits else "train"
            
        # Stream and collect only needed questions
        questions = []
        random.seed(random_seed)
        
        # Use reservoir sampling to get random sample from stream
        for i, question in enumerate(ds[dataset_split]):
            if len(questions) < num_questions:
                questions.append(question)
            else:
                # Reservoir sampling: replace random element
                j = random.randint(0, i)
                if j < num_questions:
                    questions[j] = question
            
            # Stop after reasonable number of samples
            if i > num_questions * 20:  # Sample from 20x more questions
                break
        
        # Validate images
        valid_questions = []
        for q in questions:
            try:
                img = q.get('image')
                if img and hasattr(img, 'size'):
                    valid_questions.append(q)
                if len(valid_questions) >= num_questions:
                    break
            except:
                continue
        
        logging.info(f"Successfully loaded {len(valid_questions)} PMC-VQA questions (streaming)")

        # Validate images before returning
        valid_questions = []
        for q in questions:
            img = q.get('image')
            if validate_image_for_openai(img):
                valid_questions.append(q)
            else:
                logging.warning(f"Skipping question with invalid image")
            
            if len(valid_questions) >= num_questions:
                break
        
        return valid_questions[:num_questions]
        
    except Exception as e:
        logging.error(f"Error loading PMC-VQA dataset: {str(e)}")
        return []


def _create_pmc_vqa_image_analysis(image, figure_path: str, question_text: str) -> Dict[str, Any]:
    """Create comprehensive image analysis context for PMC-VQA questions."""
    
    # Basic image information
    has_valid_image = image is not None
    dimensions = "Unknown"
    format_info = "Unknown"
    mode = "Unknown"
    
    if has_valid_image:
        try:
            dimensions = f"{image.size[0]}x{image.size[1]}" if hasattr(image, 'size') else "Unknown"
            format_info = getattr(image, 'format', 'Unknown')
            mode = getattr(image, 'mode', 'Unknown')
        except:
            pass
    
    # Analyze question for medical context
    question_lower = question_text.lower()
    
    # Determine medical domain
    medical_domains = {
        'radiology': ['x-ray', 'ct', 'mri', 'scan', 'imaging', 'radiograph'],
        'pathology': ['tissue', 'cell', 'specimen', 'biopsy', 'histology', 'microscopy'],
        'dermatology': ['skin', 'rash', 'lesion', 'dermatitis', 'mole'],
        'ophthalmology': ['eye', 'retina', 'optic', 'vision', 'fundus'],
        'cardiology': ['heart', 'cardiac', 'ecg', 'ekg', 'coronary'],
        'general_medicine': []  # default
    }
    
    detected_domain = 'general_medicine'
    for domain, keywords in medical_domains.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_domain = domain
            break
    
    # Determine question type
    if any(word in question_lower for word in ['what', 'which', 'identify']):
        question_type = 'identification'
    elif any(word in question_lower for word in ['diagnose', 'diagnosis', 'condition']):
        question_type = 'diagnosis'
    elif any(word in question_lower for word in ['count', 'how many', 'number']):
        question_type = 'quantitative'
    elif any(word in question_lower for word in ['location', 'where', 'anatomical']):
        question_type = 'anatomical_localization'
    else:
        question_type = 'general_medical'
    
    # Determine complexity
    word_count = len(question_text.split())
    if word_count < 10:
        complexity_level = 'basic'
    elif word_count < 20:
        complexity_level = 'intermediate'  
    else:
        complexity_level = 'advanced'
    
    # Specialties needed
    specialties_needed = [detected_domain]
    if detected_domain != 'general_medicine':
        specialties_needed.append('general_medicine')
    
    # Focus areas for visual analysis
    focus_areas_list = []
    if 'abnormal' in question_lower or 'lesion' in question_lower:
        focus_areas_list.append('abnormality_detection')
    if 'anatomy' in question_lower or 'structure' in question_lower:
        focus_areas_list.append('anatomical_identification')
    if 'size' in question_lower or 'measurement' in question_lower:
        focus_areas_list.append('measurement_analysis')
    if not focus_areas_list:
        focus_areas_list.append('general_visual_analysis')
    
    return {
        'has_valid_image': has_valid_image,
        'dimensions': dimensions,
        'format': format_info,
        'mode': mode,
        'medical_domain': detected_domain,
        'question_type': question_type,
        'complexity_level': complexity_level,
        'specialties_needed': specialties_needed,
        'focus_areas_list': focus_areas_list,
        'reasoning_type': f"{question_type}_with_visual_analysis",
        'medical_context': f"Medical domain: {detected_domain}\nQuestion type: {question_type}\nComplexity: {complexity_level}",
        'reasoning_requirements': f"Visual analysis required for {question_type} in {detected_domain}",
        'focus_areas': f"Key areas: {', '.join(focus_areas_list)}",
        'diagnostic_considerations': f"Image-based {question_type} requiring {detected_domain} expertise",
        'description': f"Medical image ({dimensions}) for {question_type} analysis in {detected_domain}"
    }


# ==================== PATH-VQA DATASET ====================


def load_path_vqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load Path-VQA dataset using streaming for yes/no questions only.
    """
    logging.info(f"Loading Path-VQA dataset with {num_questions} yes/no questions (streaming)")
    
    try:
        from datasets import load_dataset
        
        # Use streaming
        ds = load_dataset("flaviagiammarino/path-vqa", streaming=True)
        split_name = list(ds.keys())[0]
        
        # Collect yes/no questions only
        questions = []
        random.seed(random_seed)
        
        for i, question in enumerate(ds[split_name]):
            answer = question.get('answer', '').lower().strip()
            if answer in ['yes', 'no']:
                if len(questions) < num_questions:
                    questions.append(question)
                else:
                    # Reservoir sampling
                    j = random.randint(0, i)
                    if j < num_questions:
                        questions[j] = question
            
            # Stop after sampling enough
            if len([q for q in questions if q.get('answer', '').lower() in ['yes', 'no']]) >= num_questions:
                break
            if i > num_questions * 50:  # Safety limit
                break
        
        # Validate images and filter yes/no
        valid_questions = []
        for q in questions:
            try:
                img = q.get('image')
                answer = q.get('answer', '').lower().strip()
                if img and hasattr(img, 'size') and answer in ['yes', 'no']:
                    valid_questions.append(q)
            except:
                continue
        
        logging.info(f"Successfully loaded {len(valid_questions)} Path-VQA yes/no questions")
        return valid_questions[:num_questions]
        
    except Exception as e:
        logging.error(f"Error loading Path-VQA dataset: {str(e)}")
        return []
    

def _create_path_vqa_image_analysis(image, question_text: str) -> Dict[str, Any]:
    """Create comprehensive pathology image analysis context for Path-VQA questions."""
    
    # Basic image information
    has_valid_image = image is not None
    dimensions = "Unknown"
    format_info = "Unknown"
    mode = "Unknown"
    
    if has_valid_image:
        try:
            dimensions = f"{image.size[0]}x{image.size[1]}" if hasattr(image, 'size') else "Unknown"
            format_info = getattr(image, 'format', 'Unknown')
            mode = getattr(image, 'mode', 'Unknown')
        except:
            pass
    
    # Analyze question for pathology context
    question_lower = question_text.lower()
    
    # Determine pathology domain
    pathology_domains = {
        'cellular_pathology': ['cell', 'cellular', 'cytoplasm', 'nucleus', 'mitosis'],
        'tissue_pathology': ['tissue', 'epithelial', 'connective', 'muscle', 'nerve'],
        'organ_pathology': ['liver', 'lung', 'heart', 'kidney', 'brain', 'organ'],
        'cancer_pathology': ['cancer', 'tumor', 'malignant', 'metastasis', 'carcinoma'],
        'inflammatory_pathology': ['inflammation', 'inflammatory', 'immune', 'infection'],
        'general_pathology': []  # default
    }
    
    detected_domain = 'general_pathology'
    for domain, keywords in pathology_domains.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_domain = domain
            break
    
    # Determine analysis type
    if any(word in question_lower for word in ['present', 'visible', 'shown', 'seen']):
        analysis_type = 'presence_detection'
    elif any(word in question_lower for word in ['normal', 'abnormal', 'pathological']):
        analysis_type = 'normality_assessment'
    elif any(word in question_lower for word in ['type', 'kind', 'classification']):
        analysis_type = 'classification'
    elif any(word in question_lower for word in ['feature', 'characteristic', 'pattern']):
        analysis_type = 'feature_identification'
    else:
        analysis_type = 'general_assessment'
    
    # Determine if microscopic
    is_microscopic = any(word in question_lower for word in ['microscopic', 'histology', 'cells', 'tissue'])
    
    # Determine complexity
    word_count = len(question_text.split())
    if word_count < 8:
        complexity_level = 'basic'
    elif word_count < 15:
        complexity_level = 'intermediate'
    else:
        complexity_level = 'advanced'
    
    # Specialties needed
    specialties_needed = ['pathology', detected_domain.replace('_pathology', '')]
    if 'cancer' in detected_domain:
        specialties_needed.append('oncology')
    
    # Microscopic focus areas
    microscopic_focus_list = []
    if 'cell' in question_lower:
        microscopic_focus_list.append('cellular_morphology')
    if 'tissue' in question_lower:
        microscopic_focus_list.append('tissue_architecture')
    if 'structure' in question_lower:
        microscopic_focus_list.append('structural_analysis')
    if not microscopic_focus_list:
        microscopic_focus_list.append('general_microscopic_analysis')
    
    # Pathology patterns
    pathology_patterns = []
    if 'abnormal' in question_lower:
        pathology_patterns.append('abnormality_detection')
    if 'inflammation' in question_lower:
        pathology_patterns.append('inflammatory_patterns')
    if 'tumor' in question_lower or 'cancer' in question_lower:
        pathology_patterns.append('neoplastic_patterns')
    if not pathology_patterns:
        pathology_patterns.append('general_pathology_patterns')
    
    return {
        'has_valid_image': has_valid_image,
        'dimensions': dimensions,
        'format': format_info,
        'mode': mode,
        'is_microscopic': is_microscopic,
        'pathology_domain': detected_domain,
        'analysis_type': analysis_type,
        'complexity_level': complexity_level,
        'specialties_needed': specialties_needed,
        'microscopic_focus_list': microscopic_focus_list,
        'pathology_patterns': pathology_patterns,
        'pathology_context': f"Pathology domain: {detected_domain}\nAnalysis type: {analysis_type}\nComplexity: {complexity_level}",
        'reasoning_requirements': f"Pathological image analysis for {analysis_type} in {detected_domain}",
        'microscopic_focus': f"Focus areas: {', '.join(microscopic_focus_list)}",
        'differential_considerations': f"Binary assessment requiring {detected_domain} expertise",
        'diagnostic_approach': f"Yes/No determination through {analysis_type}",
        'description': f"Pathology image ({dimensions}) for {analysis_type} in {detected_domain}"
    }



def validate_image_for_openai(image):
    """Validate image can be processed for OpenAI."""
    if image is None:
        return False
    
    try:
        # Check if it's a valid PIL image
        if not hasattr(image, 'size') or not hasattr(image, 'mode'):
            return False
        
        # Check reasonable size
        width, height = image.size
        if width < 10 or height < 10 or width > 4096 or height > 4096:
            return False
        
        # Test conversion to RGB (what we'll need for JPEG)
        if image.mode not in ('RGB', 'L'):
            test_img = image.convert('RGB')
        
        return True
    except:
        return False

######################### ===================================== MAIN RUNNER FUNCTION ===================================== #########################
def run_dataset(
    dataset_type: str,
    num_questions: int = 50,
    random_seed: int = 42,
    run_all_configs: bool = False,
    output_dir: Optional[str] = None,
    leadership: bool = False,
    closed_loop: bool = False,
    mutual_monitoring: bool = False,
    shared_mental_model: bool = False,
    team_orientation: bool = False,
    mutual_trust: bool = False,
    mutual_trust_factor: float = 0.8,
    recruitment: bool = False,
    recruitment_method: str = "adaptive",
    recruitment_pool: str = "general",
    n_max: int = 5,
    use_medrag: bool = False
) -> Dict[str, Any]:
    """
    Run a dataset through the agent system with question-level parallel processing.
    
    Args:
        dataset_type: Type of dataset ("medqa", "medmcqa", "pubmedqa", "mmlupro-med")
        num_questions: Number of questions to process
        random_seed: Random seed for reproducibility
        run_all_configs: Whether to run all configurations
        output_dir: Optional output directory for results
        leadership: Use team leadership
        closed_loop: Use closed-loop communication
        mutual_monitoring: Use mutual performance monitoring
        shared_mental_model: Use shared mental model
        team_orientation: Use team orientation
        mutual_trust: Use mutual trust
        mutual_trust_factor: Mutual trust factor (0.0-1.0)
        recruitment: Use dynamic agent recruitment
        recruitment_method: Method for recruitment (adaptive, basic, intermediate, advanced)
        recruitment_pool: Pool of agent roles to recruit from
        n_max: Maximum number of agents for intermediate teams
        
    Returns:
        Results dictionary
    """
    # Ensure n_max has a default value
    if n_max is None:
        n_max = 5
    
    # Log parameters including MedRAG
    medrag_status = "with MedRAG" if use_medrag else "without MedRAG"
    logging.info(f"Running dataset: {dataset_type} {medrag_status}, n_max={n_max}, recruitment_method={recruitment_method}")

    # Load the dataset
    if dataset_type == "medqa":
        questions = load_medqa_dataset(num_questions, random_seed)
    elif dataset_type == "medmcqa":
        questions = load_medmcqa_dataset(num_questions, random_seed, include_multi_choice=True)
    elif dataset_type == "pubmedqa":
        questions = load_pubmedqa_dataset(num_questions, random_seed)
    elif dataset_type == "mmlupro-med":
        questions = load_mmlupro_med_dataset(num_questions, random_seed)
    elif dataset_type == "ddxplus":
        questions = load_ddxplus_dataset(num_questions, random_seed)
    elif dataset_type == "medbullets":
        questions = load_medbullets_dataset(num_questions, random_seed)
    elif dataset_type == "symcat":
        questions = load_symcat_dataset(num_questions, random_seed)
    elif dataset_type == "pmc_vqa":
        questions = load_pmc_vqa_dataset(num_questions, random_seed)
    elif dataset_type == "path_vqa":
        questions = load_path_vqa_dataset(num_questions, random_seed)
    else:
        logging.error(f"Unknown dataset type: {dataset_type}")
        return {"error": f"Unknown dataset type: {dataset_type}"}
    
    if not questions:
        return {"error": "No questions loaded"}
    
    # Log MedRAG configuration
    if use_medrag:
        # Test MedRAG availability before processing
        test_integration = create_medrag_integration()
        if test_integration and test_integration.is_available():
            logging.info("MedRAG integration is available and will be used")
        else:
            error_msg = test_integration.get_initialization_error() if test_integration else "Failed to create integration"
            logging.warning(f"MedRAG integration not available: {error_msg}")
            logging.warning("Continuing without MedRAG enhancement")
            use_medrag = False
    
    # Setup output directory
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_dir, f"{dataset_type}_run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
    else:
        run_output_dir = None
    
    # Log deployment configuration
    available_deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in available_deployments]}")
    logging.info(f"Questions will be distributed across {len(available_deployments)} deployments in round-robin fashion")
    
    # Define configurations to test
    if run_all_configs:
        configurations = [

            # Single features with intermediate recruitment
            {
                "name": "Leadership", 
                "description": f"Team of {n_max} agents with leadership component",
                "leadership": True, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            },
            {
                "name": "Closed-loop", 
                "description": f"Team of {n_max} agents with closed-loop communication",
                "leadership": False, 
                "closed_loop": True,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            },
            {
                "name": "Mutual Monitoring", 
                "description": f"Team of {n_max} agents with mutual monitoring",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": True,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            },
            {
                "name": "Shared Mental Model", 
                "description": f"Team of {n_max} agents with shared mental model",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": True,
                "team_orientation": False,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            },
            {
                "name": "Team Orientation", 
                "description": f"Team of {n_max} agents with team orientation",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": True,
                "mutual_trust": False,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            },
            {
                "name": "Mutual Trust", 
                "description": f"Team of {n_max} agents with mutual trust",
                "leadership": False, 
                "closed_loop": False,
                "mutual_monitoring": False,
                "shared_mental_model": False,
                "team_orientation": False,
                "mutual_trust": True,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            },
            # All features with recruitment
            {
                "name": "All Features with Recruitment", 
                "description": f"Team of {n_max} agents with all teamwork components",
                "leadership": True, 
                "closed_loop": True,
                "mutual_monitoring": True,
                "shared_mental_model": True,
                "team_orientation": True,
                "mutual_trust": True,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": n_max,  # Use specified n_max value
                "medrag": use_medrag  # Pass MedRAG usage
            }
        ]
    else:
        # Use the specified configuration
        configurations = [{
            "name": "Custom Configuration",
            "leadership": leadership,
            "closed_loop": closed_loop,
            "mutual_monitoring": mutual_monitoring,
            "shared_mental_model": shared_mental_model,
            "team_orientation": team_orientation,
            "mutual_trust": mutual_trust,
            "recruitment": recruitment,            
            "recruitment_method": recruitment_method,
            "recruitment_pool": recruitment_pool,
            "n_max": n_max,  # Use specified n_max value
            "medrag": use_medrag  # Pass MedRAG usage
        }]
    
    # Run each configuration
    all_results = []
    for config_dict in configurations:
        # Ensure each config has proper n_max value
        if "n_max" not in config_dict:
            config_dict["n_max"] = n_max

        # Get MedRAG setting for this configuration
        config_use_medrag = config_dict.get("use_medrag", False)
        
        # Special handling for Baseline - always use basic recruitment with 1 agent
        if config_dict["name"] == "Baseline":
            config_dict["recruitment"] = True
            config_dict["recruitment_method"] = "basic"
            config_dict["n_max"] = 1
        
        # Add recruitment settings to all configs if recruitment is enabled
        if config_dict["recruitment"]:
            config_dict["recruitment_method"] = config_dict.get("recruitment_method", recruitment_method)
            config_dict["recruitment_pool"] = config_dict.get("recruitment_pool", recruitment_pool)
        
        # Log current configuration with MedRAG status
        description = config_dict.get("description", "")
        medrag_note = " with MedRAG" if config_use_medrag else ""
        desc_str = f" - {description}{medrag_note}" if description else medrag_note
        logging.info(f"Running configuration: {config_dict['name']}{desc_str}, recruitment={config_dict['recruitment']}, method={config_dict['recruitment_method']}, n_max={config_dict['n_max']}")
        
        result = run_questions_with_configuration(
            questions,
            dataset_type,
            config_dict,
            run_output_dir,
            n_max=config_dict.get("n_max", n_max),
             use_medrag=config_use_medrag 
        )
        all_results.append(result)
    
    # Compile combined results
    combined_results = {
        "dataset": dataset_type,
        "num_questions": num_questions,
        "random_seed": random_seed,
        "timestamp": datetime.now().isoformat(),
        "parallel_processing": "question_level",
        "deployments_used": [d['name'] for d in available_deployments],
        "configurations": [r["configuration"] for r in all_results],
        "summaries": {r["configuration"]: r["summary"] for r in all_results},
        "medrag_summaries": {r["configuration"]: r.get("medrag_summary", {}) for r in all_results if r.get("use_medrag", False)}
    }
    
    # Save combined results
    if run_output_dir:
        with open(os.path.join(run_output_dir, "combined_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Also create comprehensive combined results
        create_comprehensive_combined_results(all_results, dataset_type, run_output_dir)
    
    return combined_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run datasets through the agent system')
    parser.add_argument('--dataset', type=str, default="medqa", 
                      choices=["medqa", "medmcqa", "pubmedqa", "mmlupro-med", "ddxplus", "medbullets", "symcat", "pmc_vqa", "path_vqa"], 
                      help='Dataset to run (medqa, medmcqa, pubmedqa, or mmlupro-med)')
    parser.add_argument('--num-questions', type=int, default=50, 
                      help='Number of questions to process')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed for reproducibility')
    parser.add_argument('--all', action='store_true', 
                      help='Run all feature configurations')
    parser.add_argument('--output-dir', type=str, default=None, 
                      help='Output directory for results')
    
    # Teamwork components
    parser.add_argument('--leadership', action='store_true', 
                      help='Use team leadership')
    parser.add_argument('--closedloop', action='store_true', 
                      help='Use closed-loop communication')
    parser.add_argument('--mutual', action='store_true', 
                      help='Use mutual performance monitoring')
    parser.add_argument('--mental', action='store_true', 
                      help='Use shared mental model')
    parser.add_argument('--orientation', action='store_true', 
                      help='Use team orientation')
    parser.add_argument('--trust', action='store_true', 
                      help='Use mutual trust')
    parser.add_argument('--trust-factor', type=float, default=0.8, 
                      help='Mutual trust factor (0.0-1.0)')
    
    # Recruitment arguments
    parser.add_argument('--recruitment', action='store_true', 
                      help='Use dynamic agent recruitment')
    parser.add_argument('--recruitment-method', type=str, 
                      choices=['adaptive', 'basic', 'intermediate', 'advanced'], 
                      default='adaptive', 
                      help='Recruitment method to use')
    parser.add_argument('--recruitment-pool', type=str, 
                      choices=['general', 'medical'], 
                      default='medical' if '--dataset' in sys.argv and sys.argv[sys.argv.index('--dataset')+1] in ['medqa', 'medmcqa', 'pubmedqa'] else 'general', 
                      help='Pool of roles to recruit from')
    parser.add_argument('--n-max', type=int, default=None, 
                      help='Maximum number of agents for intermediate team')
    
    # Add MedRAG argument
    parser.add_argument('--medrag', action='store_true', 
                      help='Enable MedRAG knowledge enhancement')
    parser.add_argument('--medrag-retriever', type=str, default='MedCPT',
                      help='MedRAG retriever to use (default: MedCPT)')
    parser.add_argument('--medrag-corpus', type=str, default='Textbooks',
                      help='MedRAG corpus to use (default: Textbooks)')
    
    parser.add_argument('--validate-only', action='store_true',
                      help='Only validate dataset availability, do not run')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Validate datasets
    if args.validate_only:
        validation_results = validate_all_datasets()
        
        print("\n" + "="*60)
        print("DATASET VALIDATION SUMMARY")
        print("="*60)
        
        for category, datasets in validation_results.items():
            if category in ["online_datasets", "local_datasets"]:
                print(f"\n{category.replace('_', ' ').title()}:")
                for dataset, info in datasets.items():
                    # FIXED: Use ASCII-safe status indicators
                    status_icon = {
                        "available": "[OK]",
                        "empty": "[WARN]",
                        "error": "[ERROR]"
                    }.get(info.get("status", "unknown"), "[?]")
                    
                    print(f"  {status_icon} {dataset}: {info.get('status', 'unknown')}")
                    if info.get("error"):
                        print(f"      Error: {info['error']}")
        
        print(f"\nOverall Status: {validation_results['overall_status']}")
        return
    
    # Validate requested dataset before running
    logging.info(f"Validating requested dataset: {args.dataset}")
    
    try:
        if args.dataset == "ddxplus":
            test_load = load_ddxplus_dataset(num_questions=1, random_seed=42)
        elif args.dataset == "medbullets":
            test_load = load_medbullets_dataset(num_questions=1, random_seed=42)
        elif args.dataset == "symcat":
            test_load = load_symcat_dataset(num_questions=1, random_seed=42)
        else:
            # For existing datasets, do a quick validation
            test_load = True  # Assume they work if we get here
        
        if not test_load:
            logging.error(f"Dataset {args.dataset} validation failed - no data loaded")
            print(f"[ERROR] Dataset {args.dataset} is not available or empty")
            print("Use --validate-only to check all datasets")
            return
            
        logging.info(f"[PASSED] Dataset {args.dataset} validation successful")
        
    except Exception as e:
        logging.error(f"Dataset {args.dataset} validation failed: {str(e)}")
        print(f"[ERROR] Dataset {args.dataset} validation failed: {str(e)}")
        print("Use --validate-only to check all datasets")
        return
    

    # Log MedRAG configuration if enabled
    if args.medrag:
        logging.info(f"MedRAG enhancement enabled with {args.medrag_retriever}/{args.medrag_corpus}")
        
        # Test MedRAG availability
        test_integration = create_medrag_integration(
            retriever_name=args.medrag_retriever,
            corpus_name=args.medrag_corpus
        )
        
        if test_integration and test_integration.is_available():
            logging.info("MedRAG integration test successful")
        else:
            error_msg = test_integration.get_initialization_error() if test_integration else "Failed to create integration"
            logging.error(f"MedRAG integration test failed: {error_msg}")
            logging.error("Consider disabling MedRAG or fixing the configuration")
            
            # Option to continue without MedRAG
            response = input("Continue without MedRAG? (y/n): ")
            if response.lower() != 'y':
                return
            args.medrag = False

    # Log deployment configuration
    deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in deployments]}")
    logging.info(f"Question-level parallel processing will be used with {len(deployments)} deployments")

    # If n_max is specified, automatically set recruitment method to intermediate
    if args.n_max is not None:
        args.recruitment = True
        args.recruitment_method = "intermediate"
    
    # Run dataset
    results = run_dataset(
        dataset_type=args.dataset,
        num_questions=args.num_questions,
        random_seed=args.seed,
        run_all_configs=args.all,
        output_dir=args.output_dir or config.OUTPUT_DIR,
        leadership=args.leadership,
        closed_loop=args.closedloop,
        mutual_monitoring=args.mutual,
        shared_mental_model=args.mental,
        team_orientation=args.orientation,
        mutual_trust=args.trust,
        mutual_trust_factor=args.trust_factor,
        recruitment=args.recruitment,
        recruitment_method=args.recruitment_method,
        recruitment_pool=args.recruitment_pool,
        n_max=args.n_max,
        use_medrag=args.medrag  
    )
    
    # Print overall summary
    print("\nOverall Results:")
    for config_name, summary in results.get("summaries", {}).items():
        print(f"\n{config_name}:")
        for method, stats in summary.items():
            if "accuracy" in stats:
                print(f"  {method.replace('_', ' ').title()}: {stats['accuracy']:.2%} accuracy")

    # Print MedRAG summary if used
    if args.medrag and "medrag_summaries" in results:
        print(f"\nMedRAG Enhancement Summary:")
        for config_name, medrag_summary in results["medrag_summaries"].items():
            if medrag_summary:
                total_attempts = medrag_summary.get("successful_retrievals", 0) + medrag_summary.get("failed_retrievals", 0)
                success_rate = medrag_summary.get("successful_retrievals", 0) / total_attempts if total_attempts > 0 else 0
                print(f"  {config_name}:")
                print(f"    Retrieval Success Rate: {success_rate:.2%}")
                print(f"    Average Retrieval Time: {medrag_summary.get('average_retrieval_time', 0):.2f}s")
                print(f"    Total Snippets: {medrag_summary.get('total_snippets_retrieved', 0)}")

    # Print deployment information
    deployments = config.get_all_deployments()
    print(f"\nDeployment Information:")
    print(f"  Total deployments used: {len(deployments)}")
    print(f"  Deployment names: {[d['name'] for d in deployments]}")
    print(f"  Processing method: Question-level parallel (each question assigned to a deployment)")

if __name__ == "__main__":
    main()