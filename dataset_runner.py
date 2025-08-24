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
from utils.token_counter import get_token_counter, reset_global_counter

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

def load_medmcqa_dataset(num_questions: int = 50, random_seed: int = 42, include_multi_choice: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load questions from the MedMCQA dataset with validation and error tracking.
    
    Returns:
        Tuple of (valid_questions, errors) where errors contains skipped questions
    """
    logging.info(f"Loading MedMCQA dataset with {num_questions} random questions")
    
    valid_questions = []
    errors = []
    
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        questions = list(ds["train"])
        
        # Log the choice type distribution for awareness
        choice_types = [q.get("choice_type", "unknown") for q in questions[:1000]]  # Sample first 1000
        choice_type_counts = {}
        for ct in choice_types:
            choice_type_counts[ct] = choice_type_counts.get(ct, 0) + 1
        
        logging.info(f"Choice type distribution in dataset sample: {choice_type_counts}")
        logging.info("Note: All questions will be treated as single-choice")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Randomly select questions (no need to filter by choice_type)
        if num_questions < len(questions):
            selected_questions = random.sample(questions, num_questions * 2)  # Get extra to account for invalid ones
        else:
            selected_questions = questions
        
        # Validate each question
        for question_data in selected_questions:
            try:
                agent_task, eval_data, is_valid = format_medmcqa_for_task(question_data)
                
                if is_valid:
                    # Store the original question data for later formatting
                    valid_questions.append(question_data)
                    if len(valid_questions) >= num_questions:
                        break
                else:
                    # eval_data contains error info when is_valid is False
                    errors.append(eval_data)
                    logging.warning(f"Skipped question {eval_data.get('question_id', 'unknown')}: {eval_data.get('message', 'Invalid')}")
                    
            except Exception as e:
                error_info = {
                    "question_id": question_data.get("id", "unknown"),
                    "error_type": "formatting_error", 
                    "message": f"Error formatting question: {str(e)}"
                }
                errors.append(error_info)
                logging.error(f"Error processing question {question_data.get('id', 'unknown')}: {str(e)}")
        
        logging.info(f"Successfully loaded {len(valid_questions)} valid questions from MedMCQA dataset")
        logging.info(f"Skipped {len(errors)} questions due to validation errors")
        
        if len(valid_questions) < num_questions:
            logging.warning(f"Only found {len(valid_questions)} valid questions, requested {num_questions}")
        
        return valid_questions[:num_questions], errors
    
    except Exception as e:
        logging.error(f"Error loading MedMCQA dataset: {str(e)}")
        return [], [{"error_type": "dataset_loading_error", "message": str(e)}]

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

def format_medmcqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Format MedMCQA question into agent task and evaluation data.
    Enhanced to validate ground truth and skip invalid questions.

    Args:
        question_data: Question data from the MedMCQA dataset.

    Returns:
        Tuple of:
        - agent_task: Task dictionary for agent system input.
        - eval_data: Ground truth, rationale, metadata for evaluation.
        - is_valid: Boolean indicating if the question should be processed
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
    
    # Parse and validate cop field
    cop = question_data.get("cop", "")
    ground_truth, is_valid_cop = parse_cop_field(cop)
    
    if not is_valid_cop:
        # Return error information for logging
        error_info = {
            "question_id": question_data.get("id", "unknown"),
            "invalid_cop": cop,
            "error_type": "invalid_ground_truth",
            "message": f"Invalid cop value '{cop}' - question skipped"
        }
        return {}, error_info, False

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

    return agent_task, eval_data, True

def parse_cop_field(cop) -> Tuple[str, bool]:
    """
    Parse the cop (correct option) field from MedMCQA dataset with 0-based indexing.
    
    Args:
        cop: The cop field value (can be int, str, or other)
        
    Returns:
        Tuple of (letter, is_valid) where:
        - letter: Single letter representing the correct answer (A, B, C, or D)
        - is_valid: Boolean indicating if the parsing was successful
    """
    # Handle integer values (most common case) - 0-based indexing
    if isinstance(cop, int):
        if 0 <= cop <= 3:
            return chr(65 + cop), True  # 0→A, 1→B, 2→C, 3→D
        else:
            return "A", False
    
    # Handle string values
    elif isinstance(cop, str):
        cop = cop.strip()
        
        # Handle numeric strings - 0-based indexing
        if cop.isdigit():
            cop_int = int(cop)
            if 0 <= cop_int <= 3:
                return chr(65 + cop_int), True
            else:
                return "A", False
        
        # Handle letter answers
        elif cop.upper() in ['A', 'B', 'C', 'D']:
            return cop.upper(), True
        
        # Handle comma-separated (take first one)
        elif ',' in cop:
            first_part = cop.split(',')[0].strip()
            return parse_cop_field(first_part)  # Recursive call
        
        # Handle empty strings
        elif not cop:
            return "A", False
        
        # Other string formats - try to extract number
        else:
            import re
            match = re.search(r'\d+', cop)
            if match:
                cop_int = int(match.group())
                if 0 <= cop_int <= 3:
                    return chr(65 + cop_int), True
                else:
                    return "A", False
            else:
                return "A", False
    
    # Handle list/array values (edge case)
    elif isinstance(cop, (list, tuple)) and len(cop) > 0:
        return parse_cop_field(cop[0])  # Take first element
    
    # Handle other types
    else:
        # Try to convert to string and parse
        try:
            return parse_cop_field(str(cop))
        except:
            return "A", False


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

# ==================== VERIFICATION FUNCTION ====================



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
                            use_medrag: bool = False,
                            validation_errors: List = None) -> Dict[str, Any]:
    """
    Enhanced question processing with robust validation and error recovery.
    Updated to handle new MedMCQA validation format.
    """
    import gc
    import traceback
    import time

    # Create unique simulation ID
    sim_id = f"{dataset_type}_{configuration['name'].lower().replace(' ', '_')}_q{question_index}_{deployment_config['name']}"
    if use_medrag:
        sim_id += "_medrag"
    
    # Format question based on dataset type with enhanced error handling
    try:
        if dataset_type == "medqa":
            agent_task, eval_data = format_medqa_for_task(question)
            is_valid = True
        elif dataset_type == "medmcqa":
            agent_task, eval_data, is_valid = format_medmcqa_for_task(question)
            if not is_valid:
                # eval_data contains error info when is_valid is False
                if validation_errors is not None:
                    validation_errors.append(eval_data)
                
                return {
                    "question_index": question_index,
                    "deployment_used": deployment_config['name'],
                    "simulation_id": sim_id,
                    "error": f"Validation error: {eval_data.get('message', 'Invalid question')}",
                    "validation_error": True,
                    "error_details": eval_data
                }
        elif dataset_type == "mmlupro-med":
            agent_task, eval_data = format_mmlupro_med_for_task(question)
            is_valid = True
        elif dataset_type == "pubmedqa":
            agent_task, eval_data = format_pubmedqa_for_task(question)
            is_valid = True
        elif dataset_type == "ddxplus":
            agent_task, eval_data = format_ddxplus_for_task(question)
            is_valid = True
        elif dataset_type == "medbullets":
            agent_task, eval_data = format_medbullets_for_task(question)
            is_valid = True
        elif dataset_type == "pmc_vqa":
            agent_task, eval_data = format_pmc_vqa_for_task(question)
            is_valid = True
        elif dataset_type == "path_vqa":
            agent_task, eval_data = format_path_vqa_for_task(question)
            is_valid = True
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    except Exception as e:
        logging.error(f"Failed to format question {question_index}: {str(e)}")
        return {
            "question_index": question_index,
            "deployment_used": deployment_config['name'],
            "simulation_id": sim_id,
            "error": f"Question formatting error: {str(e)}",
            "format_error": True
        }

    # Set thread-local data for this question
    thread_local.question_index = question_index
    thread_local.question_task = agent_task
    thread_local.question_eval = eval_data

    # Initialize result structure with enhanced vision metadata
    question_result = {
        "question_index": question_index,
        "deployment_used": deployment_config['name'],
        "simulation_id": sim_id,
        "dataset_type": dataset_type,
        "has_image": agent_task.get("image_data", {}).get("image_available", False),
        "image_type": agent_task.get("image_data", {}).get("image_type", "none"),
        "requires_vision": agent_task.get("image_data", {}).get("requires_visual_analysis", False),
        "recruitment_info": {},
        "agent_responses": {},
        "disagreement_analysis": {},
        "disagreement_flag": False,
        "medrag_info": {},
        "vision_performance": {}
    }

    # Enhanced retry logic with vision-specific error handling
    for attempt in range(max_retries):
        try:
            # Log attempt with vision status
            vision_status = "with vision" if question_result["has_image"] else "text-only"
            medrag_status = "with MedRAG" if use_medrag else "without MedRAG"
            logging.info(f"Processing Q{question_index} attempt {attempt+1}/{max_retries} "
                        f"({vision_status}, {medrag_status}) on {deployment_config['name']}")

            # Create simulator with enhanced configuration
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
                n_max=configuration.get("n_max", 4),
                deployment_config=deployment_config,
                question_specific_context=True,
                task_config=agent_task,
                eval_data=eval_data,
                use_medrag=use_medrag
            )

            # Initialize token tracking for this question
            token_counter = get_token_counter()
            pre_simulation_usage = token_counter.get_session_usage()
            
            # Run simulation with enhanced error capture
            simulation_results = simulator.run_simulation()
            performance = simulator.evaluate_performance()
            
            # Get token usage for this question
            post_simulation_usage = token_counter.get_session_usage()
            question_token_usage = {
                "input_tokens": post_simulation_usage["total_usage"]["input_tokens"] - pre_simulation_usage["total_usage"]["input_tokens"],
                "output_tokens": post_simulation_usage["total_usage"]["output_tokens"] - pre_simulation_usage["total_usage"]["output_tokens"],
                "total_tokens": post_simulation_usage["total_usage"]["total_tokens"] - pre_simulation_usage["total_usage"]["total_tokens"],
                "api_calls": post_simulation_usage["total_usage"]["api_calls"] - pre_simulation_usage["total_usage"]["api_calls"]
            }
            
            # Add timing information
            pre_timing = pre_simulation_usage.get("timing_stats", {})
            post_timing = post_simulation_usage.get("timing_stats", {})
            question_timing = {
                "response_time_ms": post_timing.get("total_response_time_ms", 0) - pre_timing.get("total_response_time_ms", 0),
                "average_response_time_ms": post_timing.get("average_response_time_ms", 0)
            }

            # Extract comprehensive results
            question_result.update({
                "agent_responses": extract_agent_responses_info(simulation_results),
                "disagreement_analysis": detect_agent_disagreement(simulation_results.get("agent_responses", {})),
                "decisions": simulation_results.get("decision_results", {}),
                "performance": performance.get("task_performance", {}),
                "agent_conversations": simulation_results.get("exchanges", []),
                "simulation_metadata": simulation_results.get("simulation_metadata", {}),
                "token_usage": question_token_usage,
                "timing_stats": question_timing
            })

            # Extract MedRAG information
            if use_medrag:
                medrag_enhancement = simulation_results.get("simulation_metadata", {}).get("medrag_enhancement", {})
                if medrag_enhancement:
                    question_result["medrag_info"] = medrag_enhancement
                    logging.info(f"Q{question_index}: MedRAG success={medrag_enhancement.get('success', False)}, "
                               f"snippets={medrag_enhancement.get('snippets_retrieved', 0)}")
                else:
                    question_result["medrag_info"] = {"enabled": True, "success": False, "error": "No enhancement data"}
            else:
                question_result["medrag_info"] = {"enabled": False}

            # Extract vision performance metrics
            if question_result["has_image"]:
                vision_stats = extract_vision_performance_metrics(simulation_results)
                question_result["vision_performance"] = vision_stats
                logging.info(f"Q{question_index}: Vision usage - {vision_stats}")

            # Check for disagreements
            if question_result["disagreement_analysis"].get("has_disagreement", False):
                question_result["disagreement_flag"] = True

            # Save individual result
            if run_output_dir:
                question_result_file = os.path.join(run_output_dir, f"question_{question_index}_result.json")
                with open(question_result_file, 'w') as f:
                    json.dump(question_result, f, indent=2, default=str)  # default=str for PIL images
                logging.info(f"Saved Q{question_index} result to {question_result_file}")
                
                # Save token usage for this question
                token_usage_file = os.path.join(run_output_dir, f"question_{question_index}_token_usage.json")
                
                # Enhanced timing calculation
                question_duration_seconds = question_timing.get("response_time_ms", 0) / 1000
                avg_time_per_call = question_timing.get("average_response_time_ms", 0) / 1000
                
                with open(token_usage_file, 'w') as f:
                    json.dump({
                        "question_index": question_index,
                        "timing_summary": {
                            "total_time_seconds": round(question_duration_seconds, 2),
                            "total_time_minutes": round(question_duration_seconds / 60, 2),
                            "average_time_per_call_seconds": round(avg_time_per_call, 2),
                            "average_time_per_call_ms": round(question_timing.get("average_response_time_ms", 0), 2),
                            "total_api_calls": question_token_usage.get("api_calls", 0)
                        },
                        "token_usage": question_token_usage,
                        "detailed_timing_stats": question_timing,
                        "saved_at": datetime.now().isoformat()
                    }, f, indent=2)
                
                # Enhanced logging with vision breakdown
                if is_vision_task and question_token_usage.get("image_tokens", 0) > 0:
                    logging.info(f"Q{question_index} vision token usage: {question_token_usage['total_tokens']} total "
                               f"({question_token_usage['text_tokens']} text + {question_token_usage['image_tokens']} image), "
                               f"{question_token_usage['vision_calls']}/{question_token_usage['api_calls']} vision calls, "
                               f"{question_token_usage['vision_percentage']:.1f}% vision tokens")
                else:
                    logging.info(f"Q{question_index} token usage: {question_token_usage['total_tokens']} total tokens "
                               f"({question_token_usage['input_tokens']} in, {question_token_usage['output_tokens']} out, "
                               f"{question_token_usage['api_calls']} calls)")

            # Success - break retry loop
            break

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Q{question_index} attempt {attempt+1} failed: {error_msg}")
            
            # Enhanced error categorization
            if "invalid image" in error_msg.lower() or "base64" in error_msg.lower():
                # Image-related error
                logging.error(f"Q{question_index}: Image processing error - {error_msg}")
                if question_result["has_image"]:
                    question_result["vision_performance"]["image_error"] = error_msg
                    # For final attempt, try without image
                    if attempt == max_retries - 1:
                        logging.warning(f"Q{question_index}: Final attempt - disabling image for fallback")
                        agent_task["image_data"]["image"] = None
                        agent_task["image_data"]["image_available"] = False
                        question_result["has_image"] = False
                        question_result["vision_fallback_used"] = True
            
            elif "medrag" in error_msg.lower():
                # MedRAG-related error
                logging.error(f"Q{question_index}: MedRAG error - {error_msg}")
                question_result["medrag_info"] = {"enabled": use_medrag, "success": False, "error": error_msg}
            
            elif "timeout" in error_msg.lower():
                # Timeout error
                logging.error(f"Q{question_index}: Timeout error - {error_msg}")
                question_result["timeout_error"] = error_msg
            
            # Exponential backoff for retries
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt * 2, 30)  # Cap at 30 seconds
                logging.info(f"Q{question_index}: Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                # Final failure
                question_result["error"] = f"All attempts failed. Last error: {error_msg}"
                question_result["final_error_type"] = categorize_error(error_msg)
                logging.error(f"Q{question_index}: Final failure after {max_retries} attempts")

    # Cleanup
    try:
        if hasattr(thread_local, 'question_task'):
            delattr(thread_local, 'question_task')
        if hasattr(thread_local, 'question_eval'):
            delattr(thread_local, 'question_eval')
        if 'simulator' in locals():
            del simulator
        gc.collect()
    except Exception as cleanup_error:
        logging.warning(f"Cleanup warning for Q{question_index}: {cleanup_error}")

    return question_result


def extract_vision_performance_metrics(simulation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract vision performance metrics including token costs."""
    vision_stats = {
        "agents_used_vision_round1": 0,
        "agents_used_vision_round3": 0,
        "vision_errors": 0,
        "vision_fallbacks": 0,
        "total_agents": 0,
        "vision_success_rate": 0.0
    }
    
    try:
        # Get agent responses for Round 3 (final decisions)
        agent_responses = simulation_results.get("agent_responses", {})
        
        for agent_role, response_data in agent_responses.items():
            vision_stats["total_agents"] += 1
            
            if isinstance(response_data, dict):
                # Check Round 1 vision usage
                if response_data.get("used_vision_round1", False):
                    vision_stats["agents_used_vision_round1"] += 1
                
                # Check Round 3 vision usage  
                if response_data.get("used_vision_round3", False):
                    vision_stats["agents_used_vision_round3"] += 1
                
                # Check for vision errors
                if "vision_error" in response_data or (
                    "error" in response_data and 
                    "image" in str(response_data.get("error", "")).lower()
                ):
                    vision_stats["vision_errors"] += 1
                
                # Check for fallback usage
                if response_data.get("vision_fallback_used", False):
                    vision_stats["vision_fallbacks"] += 1
        
        # Calculate success rate based on agents that successfully used vision
        successful_vision = max(
            vision_stats["agents_used_vision_round1"], 
            vision_stats["agents_used_vision_round3"]
        ) - vision_stats["vision_errors"]
        
        if vision_stats["total_agents"] > 0:
            vision_stats["vision_success_rate"] = successful_vision / vision_stats["total_agents"]
        
    except Exception as e:
        logging.warning(f"Error extracting vision metrics: {e}")
        vision_stats["extraction_error"] = str(e)
    
    return vision_stats


def categorize_error(error_msg: str) -> str:
    """Categorize error for better debugging."""
    error_lower = error_msg.lower()
    
    if "image" in error_lower or "base64" in error_lower or "vision" in error_lower:
        return "vision_error"
    elif "medrag" in error_lower:
        return "medrag_error"
    elif "timeout" in error_lower:
        return "timeout_error"
    elif "deployment" in error_lower or "api" in error_lower:
        return "api_error"
    elif "recruitment" in error_lower:
        return "recruitment_error"
    else:
        return "unknown_error"


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


# ==================== IMAGE DATASET  ====================
# ==================== PATH-VQA DATASET ====================
# ==================== PMC-VQA DATASET ====================

def validate_image_for_vision_api(image) -> bool:
    """Validate image compatibility with vision API."""
    if image is None:
        return False
    
    try:
        # Check if it's a valid PIL image
        if not hasattr(image, 'size') or not hasattr(image, 'mode'):
            return False
        
        # Check dimensions
        width, height = image.size
        if width < 10 or height < 10:
            return False
        
        # Check if dimensions are reasonable (not too large)
        if width > 4096 or height > 4096:
            return False
        
        # Test conversion to RGB
        if image.mode not in ('RGB', 'L', 'RGBA'):
            try:
                test_img = image.convert('RGB')
            except:
                return False
        
        return True
    except Exception:
        return False

def load_pmc_vqa_dataset(num_questions: int = 50, random_seed: int = 42, 
                        dataset_split: str = "test") -> List[Dict[str, Any]]:
    """
    Load PMC-VQA dataset with proper image validation.
    """
    logging.info(f"Loading PMC-VQA dataset with {num_questions} questions from {dataset_split} split")
    
    try:
        from datasets import load_dataset
        
        # Use streaming to avoid memory issues
        ds = load_dataset("hamzamooraj99/PMC-VQA-1", streaming=True)
        
        available_splits = list(ds.keys())
        logging.info(f"PMC-VQA available splits: {available_splits}")
        
        if dataset_split not in available_splits:
            dataset_split = available_splits[0] if available_splits else "train"
            logging.warning(f"Requested split not found, using {dataset_split}")
        
        # Collect valid questions with images
        questions = []
        random.seed(random_seed)
        
        attempted = 0
        max_attempts = num_questions * 10  # Try more samples to find valid ones
        
        for question in ds[dataset_split]:
            attempted += 1
            
            try:
                # Validate image first
                img = question.get('image')
                if not validate_image_for_vision_api(img):
                    continue
                
                # Check required fields
                question_text = question.get("Question", "").strip()
                if not question_text:
                    continue
                
                # Check if we have choices
                has_choices = any(question.get(f"Choice {chr(65+i)}", "").strip() 
                                for i in range(4))
                if not has_choices:
                    continue
                
                questions.append(question)
                
                if len(questions) >= num_questions:
                    break
                    
            except Exception as e:
                logging.debug(f"Skipped PMC-VQA question due to error: {e}")
                continue
            
            if attempted >= max_attempts:
                logging.warning(f"Reached max attempts ({max_attempts}), stopping with {len(questions)} questions")
                break
        
        logging.info(f"Successfully loaded {len(questions)} PMC-VQA questions with valid images")
        return questions[:num_questions]
        
    except Exception as e:
        logging.error(f"Error loading PMC-VQA dataset: {str(e)}")
        return []

def load_path_vqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load Path-VQA dataset with robust filtering to ensure ONLY yes/no questions.
    Pre-filters larger pool to guarantee exact count of valid yes/no questions.
    """
    logging.info(f"Loading Path-VQA dataset - requesting {num_questions} yes/no questions")
    
    try:
        from datasets import load_dataset
        import random
        
        # Use streaming
        ds = load_dataset("flaviagiammarino/path-vqa", streaming=True)
        split_name = list(ds.keys())[0]
        
        # PHASE 1: Collect larger pool of yes/no candidates
        candidate_pool = []
        pool_target = num_questions * 3  # Collect 3x to ensure enough after validation
        attempted = 0
        max_search_limit = num_questions * 10  # Reasonable search limit
        
        logging.info(f"Phase 1: Collecting {pool_target} yes/no candidate questions...")
        
        for question in ds[split_name]:
            attempted += 1
            
            try:
                # STRICT yes/no filtering
                answer = question.get('answer', '').lower().strip()
                if answer not in ['yes', 'no']:
                    continue
                    
                # Basic validation
                question_text = question.get('question', '').strip()
                if not question_text:
                    continue
                    
                img = question.get('image')
                if img is None:
                    continue
                
                candidate_pool.append(question)
                
                if len(candidate_pool) >= pool_target:
                    break
                    
            except Exception as e:
                logging.debug(f"Skipped candidate due to error: {e}")
                continue
            
            if attempted >= max_search_limit:
                logging.warning(f"Reached search limit ({max_search_limit}), found {len(candidate_pool)} candidates")
                break
        
        logging.info(f"Phase 1 complete: Found {len(candidate_pool)} yes/no candidates from {attempted} questions")
        
        if len(candidate_pool) < num_questions:
            logging.error(f"INSUFFICIENT YES/NO QUESTIONS: Only {len(candidate_pool)} found, need {num_questions}")
            logging.error("PathVQA dataset may not have enough yes/no questions. Reduce --num-questions parameter.")
            return candidate_pool  # Return what we have
        
        # PHASE 2: Final validation and image verification
        logging.info(f"Phase 2: Validating images for {num_questions} final questions...")
        
        # Shuffle the candidate pool
        random.seed(random_seed)
        random.shuffle(candidate_pool)
        
        final_questions = []
        validation_attempts = 0
        
        for question in candidate_pool:
            validation_attempts += 1
            
            try:
                # Re-validate answer (double-check)
                answer = question.get('answer', '').lower().strip()
                if answer not in ['yes', 'no']:
                    logging.warning(f"Validation failed: Answer '{answer}' is not yes/no")
                    continue
                
                # Thorough image validation
                img = question.get('image')
                if not validate_image_for_vision_api(img):
                    continue
                
                # Final question text check
                question_text = question.get('question', '').strip()
                if not question_text or len(question_text) < 10:
                    continue
                
                final_questions.append(question)
                
                if len(final_questions) >= num_questions:
                    break
                    
            except Exception as e:
                logging.debug(f"Validation error: {e}")
                continue
        
        logging.info(f"Phase 2 complete: Validated {len(final_questions)} questions with valid images")
        
        # PHASE 3: Final verification
        if len(final_questions) < num_questions:
            logging.warning(f"After validation: Only {len(final_questions)} questions available (requested {num_questions})")
        
        # Verify all answers are yes/no
        answer_verification = {}
        for q in final_questions:
            ans = q.get('answer', '').lower().strip()
            answer_verification[ans] = answer_verification.get(ans, 0) + 1
        
        logging.info(f"Final answer distribution: {answer_verification}")
        
        # Double-check: Remove any non yes/no that slipped through
        verified_questions = []
        for q in final_questions:
            ans = q.get('answer', '').lower().strip()
            if ans in ['yes', 'no']:
                verified_questions.append(q)
            else:
                logging.warning(f"Removing question with answer '{ans}' - not yes/no")
        
        logging.info(f"SUCCESS: Loaded {len(verified_questions)} PURE yes/no Path-VQA questions")
        return verified_questions[:num_questions]
        
    except Exception as e:
        logging.error(f"Error loading Path-VQA dataset: {str(e)}")
        return []


def format_pmc_vqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format PMC-VQA with simple Answer_label usage."""
    question_text = question_data.get("Question", "")
    answer = question_data.get("Answer", "")
    answer_label = question_data.get("Answer_label", "")
    image = question_data.get("image")
    
    # Validate image
    image_valid = validate_image_for_vision_api(image)
    if not image_valid:
        logging.warning("PMC-VQA question has invalid image, setting image to None")
        image = None
    
    # Extract choices
    choices = []
    for i, key in enumerate(["Choice A", "Choice B", "Choice C", "Choice D"]):
        choice_text = question_data.get(key, "")
        if choice_text and choice_text.strip():
            choices.append(f"{chr(65+i)}. {choice_text}")
    
    # FIXED: Use Answer_label directly as ground truth
    correct_letter = answer_label.strip().upper() if answer_label else "A"
    
    # Validate it's a proper option
    if correct_letter not in "ABCD":
        logging.warning(f"Invalid answer_label '{answer_label}', defaulting to A")
        correct_letter = "A"
    
    # Enhanced description for medical image analysis
    enhanced_description = f"""MEDICAL IMAGE ANALYSIS QUESTION

Question: {question_text}

Instructions: Carefully examine the provided medical image and use your visual analysis to answer this question. Consider:
- Anatomical structures visible in the image
- Any pathological changes or abnormalities
- Relevant clinical features
- Integration of visual findings with medical knowledge

Provide your analysis and select the most appropriate answer."""
    
    agent_task = {
        "name": "PMC-VQA Medical Image Question",
        "description": enhanced_description,
        "type": "mcq",
        "options": choices,
        "expected_output_format": f"Single letter (A-{chr(64+len(choices))}) with detailed image analysis and medical reasoning",
        "image_data": {
            "image": image,
            "image_available": image is not None,
            "requires_visual_analysis": True,
            "image_type": "medical_image"
        }
    }
    
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {correct_letter: answer},
        "metadata": {
            "dataset": "pmc_vqa", 
            "has_image": image is not None,
            "original_answer": answer,
            "original_answer_label": answer_label,
            "image_validated": image_valid,
            "num_choices": len(choices)
        }
    }
    
    return agent_task, eval_data


def format_path_vqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format Path-VQA as binary MCQ with enhanced pathology context."""
    question_text = question_data.get("question", "")
    answer = question_data.get("answer", "").lower().strip()
    image = question_data.get("image")
    
    # Validate image
    image_valid = validate_image_for_vision_api(image)
    if not image_valid:
        logging.warning("Path-VQA question has invalid image, setting image to None")
        image = None
    
    choices = ["A. Yes", "B. No"]
    correct_letter = "A" if answer == "yes" else "B"
    
    # Create enhanced pathology context
    enhanced_description = f"""PATHOLOGY IMAGE ANALYSIS QUESTION

Question: {question_text}

Instructions: Examine the provided pathology/histology image carefully. This is a microscopic image that requires detailed visual analysis. Consider:
- Cellular morphology and architecture
- Tissue patterns and organization
- Pathological changes or features
- Staining patterns and characteristics
- Integration with pathological knowledge

Based on your visual examination, answer: You must respond with option A or B, not yes/no."""
    
    agent_task = {
        "name": "Path-VQA Pathology Question",
        "description": enhanced_description,
        "type": "mcq",
        "options": choices,
        "expected_output_format": "A for Yes, B for No with detailed pathological analysis and reasoning",
        "image_data": {
            "image": image,
            "image_available": image is not None,
            "is_pathology_image": True,
            "requires_visual_analysis": True,
            "image_type": "pathology_slide"
        }
    }
    
    eval_data = {
        "ground_truth": correct_letter,
        "rationale": {correct_letter: f"Pathology analysis: {answer.title()}"},
        "metadata": {
            "dataset": "path_vqa", 
            "original_answer": answer, 
            "has_image": image is not None,
            "image_validated": image_valid
        }
    }
    
    return agent_task, eval_data 

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
    n_max: int = None,  # CHANGED: Default to None
    enable_dynamic_selection: bool = True,
    use_medrag: bool = False
) -> Dict[str, Any]:
    
    # Initialize token tracking for this run
    reset_global_counter()  # Reset global counter for fresh run
    token_counter = get_token_counter(
        output_dir=os.path.join(output_dir or "results", "token_logs") if output_dir else "token_logs",
        max_input_tokens=config.MAX_INPUT_TOKENS,
        max_output_tokens=config.MAX_OUTPUT_TOKENS
    )
    run_start_time = datetime.now()
    
    # FIXED: Only disable dynamic selection if there are explicit configs AND dynamic selection is not explicitly requested
    explicit_teamwork_config = any([leadership, closed_loop, mutual_monitoring, shared_mental_model, team_orientation, mutual_trust])
    explicit_team_size = n_max is not None
    
    # Don't auto-disable dynamic selection - respect the explicit enable_dynamic_selection parameter
    if enable_dynamic_selection:
        logging.info("Dynamic selection ENABLED - team size and teamwork components will be determined per question") 
        # Enable recruitment by default for dynamic selection
        if not recruitment:
            recruitment = True
            logging.info("Recruitment automatically enabled for dynamic selection")
    else:
        logging.info("Dynamic selection DISABLED - using provided/default configuration")
    
    # Set default n_max if not provided (2-4 range)
    if n_max is None:
        n_max = 4
    
    # Log parameters including dynamic selection
    dynamic_status = "with dynamic selection" if enable_dynamic_selection else "with static configuration"
    logging.info(f"Running dataset: {dataset_type} {dynamic_status}, n_max={n_max}, recruitment_method={recruitment_method}")

    # Load the dataset with validation error tracking (unchanged)
    validation_errors = []
    
    if dataset_type == "medqa":
        questions = load_medqa_dataset(num_questions, random_seed)
    elif dataset_type == "medmcqa":
        questions, validation_errors = load_medmcqa_dataset(num_questions, random_seed, include_multi_choice=True)
        if validation_errors:
            logging.warning(f"MedMCQA: {len(validation_errors)} questions had validation errors and were skipped")
    elif dataset_type == "pubmedqa":
        questions = load_pubmedqa_dataset(num_questions, random_seed)
    elif dataset_type == "mmlupro-med":
        questions = load_mmlupro_med_dataset(num_questions, random_seed)
    elif dataset_type == "ddxplus":
        questions = load_ddxplus_dataset(num_questions, random_seed)
    elif dataset_type == "medbullets":
        questions = load_medbullets_dataset(num_questions, random_seed)
    elif dataset_type == "pmc_vqa":
        questions = load_pmc_vqa_dataset(num_questions, random_seed)
    elif dataset_type == "path_vqa":
        questions = load_path_vqa_dataset(num_questions, random_seed)
    else:
        logging.error(f"Unknown dataset type: {dataset_type}")
        return {"error": f"Unknown dataset type: {dataset_type}"}
    
    if not questions:
        error_msg = "No questions loaded"
        if validation_errors:
            error_msg += f" (all {len(validation_errors)} questions had validation errors)"
        return {"error": error_msg, "validation_errors": validation_errors}
    
    # Log successful loading
    logging.info(f"Successfully loaded {len(questions)} valid questions")
    if validation_errors:
        logging.info(f"Skipped {len(validation_errors)} questions due to validation errors")
    
    # Setup output directory (unchanged)
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_dir, f"{dataset_type}_run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        if validation_errors:
            validation_errors_file = os.path.join(run_output_dir, "dataset_validation_errors.json")
            with open(validation_errors_file, 'w') as f:
                json.dump(validation_errors, f, indent=2)
            logging.info(f"Saved {len(validation_errors)} validation errors to {validation_errors_file}")
    else:
        run_output_dir = None
    
    # Log deployment configuration (unchanged)
    available_deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in available_deployments]}")
    logging.info(f"Questions will be distributed across {len(available_deployments)} deployments in round-robin fashion")
    
    # ENHANCED: Define configurations with dynamic selection support
    if run_all_configs:
        configurations = []
        
        if enable_dynamic_selection:
            # NEW: Dynamic configuration that adapts per question
            configurations.append({
                "name": "Dynamic Configuration", 
                "description": f"Dynamic team size (2-5) and teamwork components (max 3) selected per question",
                "leadership": None,  # Will be determined dynamically
                "closed_loop": None,
                "mutual_monitoring": None,
                "shared_mental_model": None,
                "team_orientation": None,
                "mutual_trust": None,
                "recruitment": True,
                "recruitment_method": "intermediate",
                "recruitment_pool": recruitment_pool,
                "n_max": None,  # Will be determined dynamically
                "enable_dynamic_selection": True
            })
        
        # Traditional configurations for comparison
        configurations.extend([
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
                "n_max": n_max,
                "enable_dynamic_selection": False
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
                "n_max": n_max,
                "enable_dynamic_selection": False
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
                "n_max": n_max,
                "enable_dynamic_selection": False
            },
            {
                "name": "All Features", 
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
                "n_max": n_max,
                "enable_dynamic_selection": False
            }
        ])
    else:
        # FIXED: Use the specified configuration with dynamic selection support
        configurations = [{
            "name": "Custom Configuration",
            "leadership": leadership if not enable_dynamic_selection else None,
            "closed_loop": closed_loop if not enable_dynamic_selection else None, 
            "mutual_monitoring": mutual_monitoring if not enable_dynamic_selection else None,
            "shared_mental_model": shared_mental_model if not enable_dynamic_selection else None,
            "team_orientation": team_orientation if not enable_dynamic_selection else None,
            "mutual_trust": mutual_trust if not enable_dynamic_selection else None,
            "recruitment": recruitment,  # Use the recruitment setting (now True for dynamic)            
            "recruitment_method": recruitment_method,
            "recruitment_pool": recruitment_pool,
            "n_max": n_max if not enable_dynamic_selection else None,  # None enables dynamic sizing
            "enable_dynamic_selection": enable_dynamic_selection
        }]
    
    # Run each configuration
    all_results = []
    for config_dict in configurations:
        # Ensure each config has proper settings
        if "n_max" not in config_dict:
            config_dict["n_max"] = n_max if not config_dict.get("enable_dynamic_selection", False) else None
        
        if "enable_dynamic_selection" not in config_dict:
            config_dict["enable_dynamic_selection"] = enable_dynamic_selection

        # Special handling for Baseline - always use basic recruitment with 1 agent
        if config_dict["name"] == "Baseline":
            config_dict["recruitment"] = True
            config_dict["recruitment_method"] = "basic"
            config_dict["n_max"] = 1
            config_dict["enable_dynamic_selection"] = False
        
        # Add recruitment settings to all configs if recruitment is enabled
        if config_dict["recruitment"]:
            config_dict["recruitment_method"] = config_dict.get("recruitment_method", recruitment_method)
            config_dict["recruitment_pool"] = config_dict.get("recruitment_pool", recruitment_pool)
        
        # Log current configuration with dynamic selection status
        description = config_dict.get("description", "")
        dynamic_note = " (with dynamic selection)" if config_dict.get("enable_dynamic_selection", False) else ""
        desc_str = f" - {description}{dynamic_note}" if description else dynamic_note
        logging.info(f"Running configuration: {config_dict['name']}{desc_str}, "
                    f"recruitment={config_dict['recruitment']}, method={config_dict['recruitment_method']}, "
                    f"n_max={config_dict['n_max']}, dynamic={config_dict.get('enable_dynamic_selection', False)}")
        
        result = run_questions_with_configuration(
            questions,
            dataset_type,
            config_dict,
            run_output_dir,
            n_max=config_dict.get("n_max", n_max),
            use_medrag=use_medrag,
            validation_errors=validation_errors,
            enable_dynamic_selection=config_dict.get("enable_dynamic_selection", False)  # NEW: Pass dynamic flag
        )
        all_results.append(result)
    
    # Compile combined results with dynamic selection metadata
    combined_results = {
        "dataset": dataset_type,
        "num_questions": num_questions,
        "num_valid_questions": len(questions),
        "num_validation_errors": len(validation_errors),
        "random_seed": random_seed,
        "timestamp": datetime.now().isoformat(),
        "parallel_processing": "question_level",
        "deployments_used": [d['name'] for d in available_deployments],
        "configurations": [r["configuration"] for r in all_results],
        "summaries": {r["configuration"]: r["summary"] for r in all_results},
        "dynamic_selection_enabled": enable_dynamic_selection,  # NEW: Track if dynamic selection was used
        "dynamic_selection_results": {},  # NEW: Will be populated with per-config dynamic results
        "validation_errors_summary": {
            "total_validation_errors": len(validation_errors),
            "error_types": {}
        }
    }
    
    # NEW: Collect dynamic selection results from each configuration
    for result in all_results:
        config_name = result["configuration"]
        if "dynamic_selection_summary" in result:
            combined_results["dynamic_selection_results"][config_name] = result["dynamic_selection_summary"]
    
    # Summarize validation error types (unchanged)
    if validation_errors:
        error_type_counts = {}
        for error in validation_errors:
            error_type = error.get("error_type", "unknown")
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        combined_results["validation_errors_summary"]["error_types"] = error_type_counts
    
    # Save combined results with dynamic selection metadata
    if run_output_dir:
        with open(os.path.join(run_output_dir, "combined_results.json"), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Also create comprehensive combined results
        create_comprehensive_combined_results(all_results, dataset_type, run_output_dir)
        
# Save enhanced run-level token usage summary
        run_duration = datetime.now() - run_start_time
        final_usage = token_counter.get_session_usage()
        
        # Check if this dataset has vision tasks
        is_vision_dataset = any("vqa" in dataset_type.lower() or "vision" in dataset_type.lower() for _ in [dataset_type])
        
        run_token_summary = {
            "run_metadata": {
                "dataset_type": dataset_type,
                "num_questions": num_questions,
                "random_seed": random_seed,
                "run_duration_seconds": run_duration.total_seconds(),
                "run_start_time": run_start_time.isoformat(),
                "run_end_time": datetime.now().isoformat(),
                "is_vision_dataset": is_vision_dataset
            },
            "total_token_usage": final_usage["total_usage"],
            "timing_summary": final_usage.get("timing_summary", {}),
            "questions_processed": len(all_results),
            "average_tokens_per_question": final_usage["total_usage"]["total_tokens"] / max(len(all_results), 1),
            "average_api_calls_per_question": final_usage["total_usage"]["api_calls"] / max(len(all_results), 1),
            "average_time_per_question_seconds": run_duration.total_seconds() / max(len(all_results), 1)
        }
        
        # Add vision breakdown if applicable
        if is_vision_dataset and final_usage["total_usage"].get("image_tokens", 0) > 0:
            vision_stats = token_counter.get_vision_statistics()
            run_token_summary["vision_analysis"] = {
                "total_image_tokens": final_usage["total_usage"].get("image_tokens", 0),
                "total_text_tokens": final_usage["total_usage"].get("text_tokens", 0),
                "vision_calls": final_usage["total_usage"].get("vision_calls", 0),
                "vision_call_percentage": vision_stats.get("vision_call_percentage", 0),
                "image_token_percentage": vision_stats.get("image_token_percentage", 0),
                "average_tokens_per_image": vision_stats.get("average_tokens_per_image", 0),
                "cost_impact_factor": final_usage["total_usage"].get("image_tokens", 0) / max(final_usage["total_usage"].get("text_tokens", 1), 1) if final_usage["total_usage"].get("text_tokens", 0) > 0 else 0
            }
        
        token_summary_file = os.path.join(run_output_dir, "run_token_summary.json")
        with open(token_summary_file, 'w') as f:
            json.dump(run_token_summary, f, indent=2)
        
        # Create token logs directory and save detailed usage
        token_logs_dir = os.path.join(run_output_dir, "token_logs")
        os.makedirs(token_logs_dir, exist_ok=True)
        token_counter.save_session_usage(f"{dataset_type}_run_{random_seed}")
        
        if is_vision_dataset and final_usage["total_usage"].get("image_tokens", 0) > 0:
            vision_stats = token_counter.get_vision_statistics()
            logging.info(f"Vision run completed: {final_usage['total_usage']['total_tokens']:,} total tokens "
                       f"({final_usage['total_usage'].get('text_tokens', 0):,} text + {final_usage['total_usage'].get('image_tokens', 0):,} image), "
                       f"{final_usage['total_usage']['api_calls']} API calls "
                       f"({final_usage['total_usage'].get('vision_calls', 0)} vision), "
                       f"avg {run_token_summary['average_tokens_per_question']:.1f} tokens/question")
            logging.info(f"Vision impact: {vision_stats.get('image_token_percentage', 0):.1f}% of total tokens from images, "
                       f"avg {vision_stats.get('average_tokens_per_image', 0):.1f} tokens/image")
        else:
            logging.info(f"Run completed: {final_usage['total_usage']['total_tokens']:,} total tokens, "
                       f"{final_usage['total_usage']['api_calls']} API calls, "
                       f"avg {run_token_summary['average_tokens_per_question']:.1f} tokens/question")

    
    # Add enhanced token summary to combined results
    if 'token_counter' in locals():
        final_usage = token_counter.get_session_usage()
        combined_results["token_usage_summary"] = {
            "total_tokens": final_usage["total_usage"]["total_tokens"],
            "total_api_calls": final_usage["total_usage"]["api_calls"],
            "input_tokens": final_usage["total_usage"]["input_tokens"], 
            "output_tokens": final_usage["total_usage"]["output_tokens"]
        }
        
        # Add vision breakdown if applicable
        if is_vision_dataset and final_usage["total_usage"].get("image_tokens", 0) > 0:
            combined_results["token_usage_summary"]["vision_breakdown"] = {
                "image_tokens": final_usage["total_usage"].get("image_tokens", 0),
                "text_tokens": final_usage["total_usage"].get("text_tokens", 0),
                "vision_calls": final_usage["total_usage"].get("vision_calls", 0),
                "vision_percentage": (final_usage["total_usage"].get("image_tokens", 0) / max(final_usage["total_usage"]["total_tokens"], 1)) * 100
            }
    
    return combined_results


def run_questions_with_configuration(
    questions: List[Dict[str, Any]],
    dataset_type: str,
    configuration: Dict[str, bool],
    output_dir: Optional[str] = None,
    max_retries: int = 3,
    n_max: int = 5,
    use_medrag: bool = False,
    validation_errors: List = None,
    enable_dynamic_selection: bool = False  # NEW: Dynamic selection flag
) -> Dict[str, Any]:
    """
    Enhanced configuration runner with dynamic selection support.
    
    NEW PARAMETERS:
        enable_dynamic_selection: Whether to enable dynamic selection for this configuration
    """
    config_name = configuration.get("name", "unknown")
    if use_medrag:
        config_name += "_medrag"
    
    # NEW: Add dynamic selection status to config name for clarity
    if enable_dynamic_selection:
        config_name += "_dynamic"
    
    logging.info(f"Running {len(questions)} questions with configuration: {config_name}")
    
    # Get all available deployments (unchanged)
    deployments = config.get_all_deployments()
    num_deployments = len(deployments)
    
    logging.info(f"Using {num_deployments} deployments for parallel question processing: {[d['name'] for d in deployments]}")
    
    # Reset complexity metrics (unchanged)
    from components import agent_recruitment
    agent_recruitment.reset_complexity_metrics()

    # Make sure configuration has proper recruitment method and n_max
    if config_name == "Baseline":
        configuration["recruitment"] = True
        configuration["recruitment_method"] = "basic"
        configuration["n_max"] = 1
        configuration["enable_dynamic_selection"] = False
    elif config_name != "Custom Configuration":
        configuration["recruitment"] = True
        configuration["recruitment_method"] = configuration.get("recruitment_method", "intermediate")
        if "n_max" not in configuration and not enable_dynamic_selection:
            configuration["n_max"] = n_max
    
    # Setup output directory (unchanged)
    run_output_dir = os.path.join(output_dir, f"{dataset_type}_{config_name.lower().replace(' ', '_')}") if output_dir else None
    if run_output_dir:
        os.makedirs(run_output_dir, exist_ok=True)
    
    # Initialize results structure with dynamic selection tracking
    results = {
        "configuration": config_name,
        "use_medrag": use_medrag,
        "dataset": dataset_type,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "configuration_details": configuration,
        "enable_dynamic_selection": enable_dynamic_selection,  # NEW: Track dynamic selection usage
        "dynamic_selection_summary": {  # NEW: Track dynamic selection results
            "enabled": enable_dynamic_selection,
            "team_sizes_selected": {},
            "teamwork_configs_selected": {},
            "selection_patterns": {}
        },
        "deployment_info": {
            "deployments_used": deployments,
            "parallel_processing": "question_level",
            "num_parallel_questions": num_deployments
        },
        "question_results": [],
        "errors": [],
        "validation_errors": validation_errors or [],
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
        "complexity_distribution": {"basic": 0, "intermediate": 0, "advanced": 0},
        "deployment_usage": {d['name']: 0 for d in deployments}
    }
    
    # Add summary for yes/no/maybe if PubMedQA (unchanged)
    if dataset_type == "pubmedqa":
        results["summary"]["yes_no_maybe_voting"] = {"correct": 0, "total": 0}
    
    # Track validation errors separately (unchanged)
    current_validation_errors = []
    
    # Process questions in parallel batches (unchanged logic, enhanced with dynamic selection tracking)
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
                    deployment_index = i % num_deployments
                    deployment_config = deployments[deployment_index]
                    
                    # Track deployment usage
                    results["deployment_usage"][deployment_config['name']] += 1
                    
                    # Submit the question processing with dynamic selection support
                    future = executor.submit(
                        process_single_question_enhanced,  # NEW: Enhanced function with dynamic selection
                        question_index,
                        question,
                        dataset_type,
                        configuration,
                        deployment_config,
                        run_output_dir,
                        max_retries,
                        use_medrag,
                        current_validation_errors,
                        enable_dynamic_selection  # NEW: Pass dynamic selection flag
                    )
                    
                    future_to_info[future] = {
                        "question_index": question_index,
                        "deployment": deployment_config['name']
                    }
                
                # Collect results as they complete (enhanced with dynamic selection tracking)
                for future in concurrent.futures.as_completed(future_to_info):
                    info = future_to_info[future]
                    try:
                        question_result = future.result()
                        
                        # Check if this was a validation error
                        if question_result.get("validation_error", False):
                            results["validation_errors"].append(question_result.get("error_details", {}))
                            logging.warning(f"Validation error for Q{info['question_index']}: {question_result.get('error', 'Unknown')}")
                            pbar.update(1)
                            continue
                        
                        results["question_results"].append(question_result)
                        
                        # NEW: Track dynamic selection results if enabled
                        if enable_dynamic_selection and "dynamic_selection" in question_result:
                            dynamic_info = question_result["dynamic_selection"]
                            
                            # Track team sizes
                            team_size = dynamic_info.get("team_size_selected")
                            if team_size:
                                results["dynamic_selection_summary"]["team_sizes_selected"][str(team_size)] = \
                                    results["dynamic_selection_summary"]["team_sizes_selected"].get(str(team_size), 0) + 1
                            
                            # Track teamwork configurations
                            teamwork_config = dynamic_info.get("teamwork_config_selected", {})
                            enabled_components = [k.replace("use_", "") for k, v in teamwork_config.items() if v]
                            config_key = ",".join(sorted(enabled_components)) if enabled_components else "none"
                            results["dynamic_selection_summary"]["teamwork_configs_selected"][config_key] = \
                                results["dynamic_selection_summary"]["teamwork_configs_selected"].get(config_key, 0) + 1
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'deployment': info['deployment'],
                            'processed': len(results["question_results"]),
                            'errors': len(results["errors"]),
                            'validation_errors': len(results["validation_errors"]),
                            'dynamic': "Yes" if enable_dynamic_selection else "No"
                        })
                        
                        # Update summary statistics (unchanged)
                        if "error" not in question_result and "performance" in question_result:
                            task_performance = question_result["performance"]
                            
                            if dataset_type == "pubmedqa":
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
                        
                        # Track disagreements (unchanged)
                        if question_result.get("disagreement_flag", False):
                            results["disagreement_summary"]["total_disagreements"] += 1
                            
                            disagreement_analysis = question_result.get("disagreement_analysis", {})
                            pattern = disagreement_analysis.get("answer_distribution", {})
                            pattern_key = "-".join(sorted(pattern.keys()))
                            if pattern_key:
                                results["disagreement_summary"]["disagreement_patterns"][pattern_key] = \
                                    results["disagreement_summary"]["disagreement_patterns"].get(pattern_key, 0) + 1
                        
                        # Track complexity (unchanged)
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
    
    # Add current validation errors to results (unchanged)
    if current_validation_errors:
        results["validation_errors"].extend(current_validation_errors)
    
    # Calculate final statistics (unchanged)
    total_processed = len([q for q in results["question_results"] if "error" not in q])
    results["disagreement_summary"]["disagreement_rate"] = (
        results["disagreement_summary"]["total_disagreements"] / total_processed
        if total_processed > 0 else 0
    )
    
    # Calculate accuracy for each method (unchanged)
    for method in results["summary"].keys():
        method_summary = results["summary"][method]
        method_summary["accuracy"] = (
            method_summary["correct"] / method_summary["total"] 
            if method_summary["total"] > 0 else 0.0
        )

    # Calculate mind change summary (unchanged)
    total_mind_changes = sum(
        q.get("mind_change_analysis", {}).get("agents_who_changed", 0) 
        for q in results["question_results"]
    )
    results["disagreement_summary"]["mind_change_summary"] = {
        "total_mind_changes": total_mind_changes,
        "average_change_rate": total_mind_changes / (total_processed * 5) if total_processed > 0 else 0
    }

    # Add complexity metrics (unchanged)
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
    
    # NEW: Log dynamic selection summary
    if enable_dynamic_selection:
        dynamic_summary = results["dynamic_selection_summary"]
        logging.info(f"Dynamic Selection Summary for {config_name}:")
        
        if dynamic_summary["team_sizes_selected"]:
            team_size_dist = dynamic_summary["team_sizes_selected"]
            logging.info(f"  Team sizes: {dict(team_size_dist)}")
        
        if dynamic_summary["teamwork_configs_selected"]:
            config_dist = dynamic_summary["teamwork_configs_selected"]
            logging.info(f"  Teamwork configurations: {dict(config_dist)}")
    
    # Save enhanced overall results (unchanged with dynamic selection additions)
    if run_output_dir:
        try:
            with open(os.path.join(run_output_dir, "summary.json"), 'w') as f:
                json.dump(results["summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "errors.json"), 'w') as f:
                json.dump(results["errors"], f, indent=2)
            
            if results["validation_errors"]:
                with open(os.path.join(run_output_dir, "validation_errors.json"), 'w') as f:
                    json.dump(results["validation_errors"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "disagreement_analysis.json"), 'w') as f:
                json.dump(results["disagreement_summary"], f, indent=2)
            
            with open(os.path.join(run_output_dir, "deployment_usage.json"), 'w') as f:
                json.dump(results["deployment_usage"], f, indent=2)
                
            # NEW: Save dynamic selection results
            if enable_dynamic_selection:
                with open(os.path.join(run_output_dir, "dynamic_selection_results.json"), 'w') as f:
                    json.dump(results["dynamic_selection_summary"], f, indent=2)
                
            # Save comprehensive detailed results
            with open(os.path.join(run_output_dir, "detailed_results_enhanced.json"), 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save enhanced summary results: {str(e)}")
    
    # Enhanced summary print with dynamic selection info
    print(f"\nSummary for {config_name} on {dataset_type}:")
    for method, stats in results["summary"].items():
        if "accuracy" in stats:
            print(f"  {method.replace('_', ' ').title()}: {stats['correct']}/{stats['total']} correct ({stats['accuracy']:.2%})")

    if results["errors"]:
        print(f"  Processing Errors: {len(results['errors'])}/{len(questions)} questions ({len(results['errors'])/len(questions):.2%})")
    
    if results["validation_errors"]:
        print(f"  Validation Errors: {len(results['validation_errors'])} questions skipped due to invalid ground truth")
    
    print(f"  Disagreements: {results['disagreement_summary']['total_disagreements']}/{total_processed} questions ({results['disagreement_summary']['disagreement_rate']:.2%})")
    
    # NEW: Print dynamic selection summary
    if enable_dynamic_selection:
        dynamic_summary = results["dynamic_selection_summary"]
        print(f"  Dynamic Selection:")
        
        team_sizes = dynamic_summary["team_sizes_selected"]
        if team_sizes:
            print(f"    Team sizes used: {dict(team_sizes)}")
        
        configs = dynamic_summary["teamwork_configs_selected"]
        if configs:
            print(f"    Teamwork configs used: {dict(configs)}")
    
    # Print complexity distribution (unchanged)
    if any(results["complexity_distribution"].values()):
        print(f"  Complexity Distribution: {dict(results['complexity_distribution'])}")
    
    # Print deployment usage (unchanged)
    print(f"  Deployment Usage: {dict(results['deployment_usage'])}")
        
    return results


def process_single_question_enhanced(question_index: int, 
                                   question: Dict[str, Any],
                                   dataset_type: str,
                                   configuration: Dict[str, Any],
                                   deployment_config: Dict[str, str],
                                   run_output_dir: str,
                                   max_retries: int = 3,
                                   use_medrag: bool = False,
                                   validation_errors: List = None,
                                   enable_dynamic_selection: bool = False) -> Dict[str, Any]:
    """
    Enhanced question processing with dynamic selection support.
    
    NEW PARAMETERS:
        enable_dynamic_selection: Whether to enable dynamic selection for this question
    """
    import gc
    import traceback
    import time

    # Create unique simulation ID
    sim_id = f"{dataset_type}_{configuration['name'].lower().replace(' ', '_')}_q{question_index}_{deployment_config['name']}"
    if use_medrag:
        sim_id += "_medrag"
    if enable_dynamic_selection:
        sim_id += "_dynamic"
    
    # Format question based on dataset type (unchanged)
    try:
        if dataset_type == "medqa":
            agent_task, eval_data = format_medqa_for_task(question)
            is_valid = True
        elif dataset_type == "medmcqa":
            agent_task, eval_data, is_valid = format_medmcqa_for_task(question)
            if not is_valid:
                if validation_errors is not None:
                    validation_errors.append(eval_data)
                
                return {
                    "question_index": question_index,
                    "deployment_used": deployment_config['name'],
                    "simulation_id": sim_id,
                    "error": f"Validation error: {eval_data.get('message', 'Invalid question')}",
                    "validation_error": True,
                    "error_details": eval_data
                }
        elif dataset_type == "mmlupro-med":
            agent_task, eval_data = format_mmlupro_med_for_task(question)
            is_valid = True
        elif dataset_type == "pubmedqa":
            agent_task, eval_data = format_pubmedqa_for_task(question)
            is_valid = True
        elif dataset_type == "ddxplus":
            agent_task, eval_data = format_ddxplus_for_task(question)
            is_valid = True
        elif dataset_type == "medbullets":
            agent_task, eval_data = format_medbullets_for_task(question)
            is_valid = True
        elif dataset_type == "pmc_vqa":
            agent_task, eval_data = format_pmc_vqa_for_task(question)
            is_valid = True
        elif dataset_type == "path_vqa":
            agent_task, eval_data = format_path_vqa_for_task(question)
            is_valid = True
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    except Exception as e:
        logging.error(f"Failed to format question {question_index}: {str(e)}")
        return {
            "question_index": question_index,
            "deployment_used": deployment_config['name'],
            "simulation_id": sim_id,
            "error": f"Question formatting error: {str(e)}",
            "format_error": True
        }

    # Set thread-local data for this question
    thread_local.question_index = question_index
    thread_local.question_task = agent_task
    thread_local.question_eval = eval_data

    # Initialize result structure with enhanced dynamic selection metadata
    question_result = {
        "question_index": question_index,
        "deployment_used": deployment_config['name'],
        "simulation_id": sim_id,
        "dataset_type": dataset_type,
        "enable_dynamic_selection": enable_dynamic_selection,  # NEW: Track if dynamic selection was enabled
        "dynamic_selection": {},  # NEW: Will store dynamic selection results
        "has_image": agent_task.get("image_data", {}).get("image_available", False),
        "image_type": agent_task.get("image_data", {}).get("image_type", "none"),
        "requires_vision": agent_task.get("image_data", {}).get("requires_visual_analysis", False),
        "recruitment_info": {},
        "agent_responses": {},
        "disagreement_analysis": {},
        "disagreement_flag": False,
        "team_composition": {},  # NEW: Track final team composition
        "vision_performance": {}
    }

    # Enhanced retry logic with dynamic selection support
    for attempt in range(max_retries):
        try:
            # Log attempt with dynamic selection status
            vision_status = "with vision" if question_result["has_image"] else "text-only"
            dynamic_status = "with dynamic selection" if enable_dynamic_selection else "static config"
            logging.info(f"Processing Q{question_index} attempt {attempt+1}/{max_retries} "
                        f"({vision_status}, {dynamic_status}) on {deployment_config['name']}")

            # ENHANCED: Create simulator with dynamic selection support
            simulator = AgentSystemSimulator(
                simulation_id=sim_id,
                use_team_leadership=configuration.get("leadership"),
                use_closed_loop_comm=configuration.get("closed_loop"),
                use_mutual_monitoring=configuration.get("mutual_monitoring"),
                use_shared_mental_model=configuration.get("shared_mental_model"),
                use_team_orientation=configuration.get("team_orientation"),
                use_mutual_trust=configuration.get("mutual_trust"),
                use_recruitment=configuration.get("recruitment", False),
                recruitment_method=configuration.get("recruitment_method", "adaptive"),
                recruitment_pool=configuration.get("recruitment_pool", "general"),
                n_max=configuration.get("n_max", 4),
                deployment_config=deployment_config,
                question_specific_context=True,
                task_config=agent_task,
                eval_data=eval_data,
                enable_dynamic_selection=enable_dynamic_selection  # NEW: Pass dynamic selection flag
            )

            # Initialize token tracking for this question
            token_counter = get_token_counter()
            pre_simulation_usage = token_counter.get_session_usage()
            
            # Run simulation with enhanced error capture
            simulation_results = simulator.run_simulation()
            performance = simulator.evaluate_performance()
            
            # Get enhanced token usage for this question
            post_simulation_usage = token_counter.get_session_usage()
            question_token_usage = {
                "input_tokens": post_simulation_usage["total_usage"]["input_tokens"] - pre_simulation_usage["total_usage"]["input_tokens"],
                "output_tokens": post_simulation_usage["total_usage"]["output_tokens"] - pre_simulation_usage["total_usage"]["output_tokens"],
                "total_tokens": post_simulation_usage["total_usage"]["total_tokens"] - pre_simulation_usage["total_usage"]["total_tokens"],
                "api_calls": post_simulation_usage["total_usage"]["api_calls"] - pre_simulation_usage["total_usage"]["api_calls"]
            }
            
            # Add vision token tracking if this is a vision task
            is_vision_task = question_result.get("has_image", False)
            if is_vision_task:
                question_token_usage.update({
                    "image_tokens": post_simulation_usage["total_usage"].get("image_tokens", 0) - pre_simulation_usage["total_usage"].get("image_tokens", 0),
                    "text_tokens": post_simulation_usage["total_usage"].get("text_tokens", 0) - pre_simulation_usage["total_usage"].get("text_tokens", 0),
                    "vision_calls": post_simulation_usage["total_usage"].get("vision_calls", 0) - pre_simulation_usage["total_usage"].get("vision_calls", 0),
                    "vision_percentage": 0.0
                })
                
                # Calculate vision percentage
                if question_token_usage["total_tokens"] > 0:
                    question_token_usage["vision_percentage"] = (question_token_usage["image_tokens"] / question_token_usage["total_tokens"]) * 100

            
            # Add timing information
            pre_timing = pre_simulation_usage.get("timing_stats", {})
            post_timing = post_simulation_usage.get("timing_stats", {})
            question_timing = {
                "response_time_ms": post_timing.get("total_response_time_ms", 0) - pre_timing.get("total_response_time_ms", 0),
                "average_response_time_ms": post_timing.get("average_response_time_ms", 0)
            }

            # Extract comprehensive results
            question_result.update({
                "agent_responses": extract_agent_responses_info(simulation_results),
                "disagreement_analysis": detect_agent_disagreement(simulation_results.get("agent_responses", {})),
                "decisions": simulation_results.get("decision_results", {}),
                "performance": performance.get("task_performance", {}),
                "agent_conversations": simulation_results.get("exchanges", []),
                "simulation_metadata": simulation_results.get("simulation_metadata", {}),
                "token_usage": question_token_usage,
                "timing_stats": question_timing
            })

            # NEW: Extract dynamic selection results
            if enable_dynamic_selection and "simulation_metadata" in simulation_results:
                sim_metadata = simulation_results["simulation_metadata"]
                if "dynamic_selection" in sim_metadata:
                    question_result["dynamic_selection"] = sim_metadata["dynamic_selection"]
                    logging.info(f"Q{question_index}: Dynamic selection results captured")
                
                # Extract team composition
                if "team_composition" in sim_metadata:
                    question_result["team_composition"] = sim_metadata["team_composition"]
            
            # Extract vision performance metrics (unchanged)
            if question_result["has_image"]:
                vision_stats = extract_vision_performance_metrics(simulation_results)
                question_result["vision_performance"] = vision_stats
                logging.info(f"Q{question_index}: Vision usage - {vision_stats}")

            # Check for disagreements (unchanged)
            if question_result["disagreement_analysis"].get("has_disagreement", False):
                question_result["disagreement_flag"] = True

            # Save individual result (unchanged)
            if run_output_dir:
                question_result_file = os.path.join(run_output_dir, f"question_{question_index}_result.json")
                with open(question_result_file, 'w') as f:
                    json.dump(question_result, f, indent=2, default=str)
                logging.info(f"Saved Q{question_index} result to {question_result_file}")
                
                # Save token usage for this question
                token_usage_file = os.path.join(run_output_dir, f"question_{question_index}_token_usage.json")
                
                # Enhanced timing calculation
                question_duration_seconds = question_timing.get("response_time_ms", 0) / 1000
                avg_time_per_call = question_timing.get("average_response_time_ms", 0) / 1000
                
                token_summary = {
                    "question_index": question_index,
                    "is_vision_task": is_vision_task,
                    "timing_summary": {
                        "total_time_seconds": round(question_duration_seconds, 2),
                        "total_time_minutes": round(question_duration_seconds / 60, 2),
                        "average_time_per_call_seconds": round(avg_time_per_call, 2),
                        "average_time_per_call_ms": round(question_timing.get("average_response_time_ms", 0), 2),
                        "total_api_calls": question_token_usage.get("api_calls", 0)
                    },
                    "token_usage": question_token_usage,
                    "detailed_timing_stats": question_timing,
                    "saved_at": datetime.now().isoformat()
                }
                
                # Add vision breakdown if applicable
                if is_vision_task and question_token_usage.get("image_tokens", 0) > 0:
                    token_summary["vision_breakdown"] = {
                        "vision_calls": question_token_usage["vision_calls"],
                        "text_only_calls": question_token_usage["api_calls"] - question_token_usage["vision_calls"],
                        "image_tokens": question_token_usage["image_tokens"],
                        "text_tokens": question_token_usage["text_tokens"],
                        "vision_token_percentage": question_token_usage["vision_percentage"],
                        "cost_impact_note": "Image tokens significantly increase costs for vision tasks"
                    }
                
                with open(token_usage_file, 'w') as f:
                    json.dump(token_summary, f, indent=2)

            # Success - break retry loop
            break

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Q{question_index} attempt {attempt+1} failed: {error_msg}")
            
            # Enhanced error categorization (unchanged)
            if "invalid image" in error_msg.lower() or "base64" in error_msg.lower():
                logging.error(f"Q{question_index}: Image processing error - {error_msg}")
                if question_result["has_image"]:
                    question_result["vision_performance"]["image_error"] = error_msg
                    if attempt == max_retries - 1:
                        logging.warning(f"Q{question_index}: Final attempt - disabling image for fallback")
                        agent_task["image_data"]["image"] = None
                        agent_task["image_data"]["image_available"] = False
                        question_result["has_image"] = False
                        question_result["vision_fallback_used"] = True
            
            elif "timeout" in error_msg.lower():
                logging.error(f"Q{question_index}: Timeout error - {error_msg}")
                question_result["timeout_error"] = error_msg
            
            # Exponential backoff for retries
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt * 2, 30)
                logging.info(f"Q{question_index}: Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                # Final failure
                question_result["error"] = f"All attempts failed. Last error: {error_msg}"
                question_result["final_error_type"] = categorize_error(error_msg)
                logging.error(f"Q{question_index}: Final failure after {max_retries} attempts")

    # Cleanup (unchanged)
    try:
        if hasattr(thread_local, 'question_task'):
            delattr(thread_local, 'question_task')
        if hasattr(thread_local, 'question_eval'):
            delattr(thread_local, 'question_eval')
        if 'simulator' in locals():
            del simulator
        gc.collect()
    except Exception as cleanup_error:
        logging.warning(f"Cleanup warning for Q{question_index}: {cleanup_error}")

    return question_result

def main():
    """Enhanced main entry point with dynamic selection support."""
    parser = argparse.ArgumentParser(description='Run datasets through the agent system with dynamic configuration')
    parser.add_argument('--dataset', type=str, default="medqa", 
                      choices=["medqa", "medmcqa", "pubmedqa", "mmlupro-med", "ddxplus", "medbullets", "pmc_vqa", "path_vqa"], 
                      help='Dataset to run')
    parser.add_argument('--num-questions', type=int, default=50, 
                      help='Number of questions to process')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed for reproducibility')
    parser.add_argument('--all', action='store_true', 
                      help='Run all feature configurations')
    parser.add_argument('--output-dir', type=str, default=None, 
                      help='Output directory for results')
    
    # Teamwork components (unchanged)
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
    
    # Recruitment arguments (unchanged)
    parser.add_argument('--recruitment', action='store_true', 
                      help='Use dynamic agent recruitment')
    parser.add_argument('--recruitment-method', type=str, 
                      choices=['adaptive', 'basic', 'intermediate', 'advanced'], 
                      default='adaptive', 
                      help='Recruitment method to use')
    parser.add_argument('--recruitment-pool', type=str, 
                      choices=['general', 'medical'], 
                      default='medical', 
                      help='Pool of roles to recruit from')
    parser.add_argument('--n-max', type=int, default=None, 
                      help='Maximum number of agents for intermediate team')
    
    # NEW: Dynamic selection arguments
    parser.add_argument('--enable-dynamic-selection', action='store_true', default=True,
                      help='Enable dynamic selection of team size and teamwork components (default: True)')
    parser.add_argument('--disable-dynamic-selection', action='store_true',
                      help='Disable dynamic selection (use static configuration)')
    
    parser.add_argument('--validate-only', action='store_true',
                      help='Only validate dataset availability, do not run')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Handle dynamic selection flags
    if args.disable_dynamic_selection:
        args.enable_dynamic_selection = False
    
    # Validate requested dataset before running (unchanged)
    logging.info(f"Validating requested dataset: {args.dataset}")
    
    try:
        if args.dataset == "ddxplus":
            test_load = load_ddxplus_dataset(num_questions=1, random_seed=42)
        elif args.dataset == "medbullets":
            test_load = load_medbullets_dataset(num_questions=1, random_seed=42)
        else:
            test_load = True
        
        if not test_load:
            logging.error(f"Dataset {args.dataset} validation failed - no data loaded")
            print(f"[ERROR] Dataset {args.dataset} is not available or empty")
            return
            
        logging.info(f"[PASSED] Dataset {args.dataset} validation successful")
        
    except Exception as e:
        logging.error(f"Dataset {args.dataset} validation failed: {str(e)}")
        print(f"[ERROR] Dataset {args.dataset} validation failed: {str(e)}")
        return

    # Log deployment configuration (unchanged)
    deployments = config.get_all_deployments()
    logging.info(f"Available deployments: {[d['name'] for d in deployments]}")
    logging.info(f"Question-level parallel processing will be used with {len(deployments)} deployments")

    # If n_max is specified, automatically set recruitment method to intermediate (unchanged)
    if args.n_max is not None:
        args.recruitment = True
        args.recruitment_method = "intermediate"
    
    # NEW: Log dynamic selection status
    if args.enable_dynamic_selection:
        logging.info("Dynamic selection ENABLED - team size and teamwork components will be determined per question")
    else:
        logging.info("Dynamic selection DISABLED - using provided/default static configuration")
    
    # Run dataset with enhanced dynamic selection support
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
        enable_dynamic_selection=args.enable_dynamic_selection,  # NEW: Pass dynamic selection flag
        use_medrag=False  # Removed MedRAG for now as requested
    )
    
    # Print overall summary with dynamic selection info
    print("\nOverall Results:")
    for config_name, summary in results.get("summaries", {}).items():
        print(f"\n{config_name}:")
        if summary and isinstance(summary, dict):
            for method, stats in summary.items():
                if stats and "accuracy" in stats:
                    print(f"  {method.replace('_', ' ').title()}: {stats['accuracy']:.2%} accuracy")
        else:
            print("  No valid summary data available")

    # NEW: Print dynamic selection summary if used
    if args.enable_dynamic_selection and "dynamic_selection_results" in results:
        print(f"\nDynamic Selection Summary:")
        dynamic_selection_results = results.get("dynamic_selection_results", {})
        if dynamic_selection_results:
            for config_name, dynamic_results in dynamic_selection_results.items():
                if dynamic_results and dynamic_results.get("enabled", False):
                    print(f"  {config_name}:")
                    
                    team_sizes = dynamic_results.get("team_sizes_selected", {})
                    if team_sizes:
                        print(f"    Team sizes used: {dict(team_sizes)}")
                    
                    configs = dynamic_results.get("teamwork_configs_selected", {})
                    if configs:
                        print(f"    Teamwork configs used: {dict(configs)}")

    # Print deployment information (unchanged)
    deployments = config.get_all_deployments()
    print(f"\nDeployment Information:")
    print(f"  Total deployments used: {len(deployments)}")
    print(f"  Deployment names: {[d['name'] for d in deployments]}")
    print(f"  Processing method: Question-level parallel (each question assigned to a deployment)")

if __name__ == "__main__":
    main()