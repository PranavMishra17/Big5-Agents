"""
Configuration settings for the modular agent system.
Updated to support question-level parallel processing.
"""
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Azure OpenAI settings - Multiple deployments for parallel processing
AZURE_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT')

# Multiple deployment configuration for question-level parallel processing
AZURE_DEPLOYMENTS = [
    {
        "name": "deployment_1",
        "deployment": "VARELab-GPT4o",
        "api_key": AZURE_API_KEY,
        "endpoint": AZURE_ENDPOINT,
        "api_version": "2024-08-01-preview"
    },
    {
        "name": "deployment_2", 
        "deployment": os.environ.get('AZURE_DEPLOYMENT_2', "VARELab-GPT4o-2"),
        "api_key": os.environ.get('AZURE_OPENAI_API_KEY', AZURE_API_KEY),
        "endpoint": os.environ.get('AZURE_ENDPOINT', AZURE_ENDPOINT),
        "api_version": "2025-01-01-preview"
    },
    {
        "name": "deployment_3", 
        "deployment": os.environ.get('AZURE_DEPLOYMENT_3', "VARELab-GPT4o-3"),
        "api_key": os.environ.get('AZURE_OPENAI_API_KEY', AZURE_API_KEY),
        "endpoint": os.environ.get('AZURE_ENDPOINT', AZURE_ENDPOINT),
        "api_version": "2025-01-01-preview"
    },
    {
        "name": "deployment_4", 
        "deployment": os.environ.get('AZURE_DEPLOYMENT_4', "VARELab-GPT4o-4"),
        "api_key": os.environ.get('AZURE_OPENAI_API_KEY', AZURE_API_KEY),
        "endpoint": os.environ.get('AZURE_ENDPOINT', AZURE_ENDPOINT),
        "api_version": "2025-01-01-preview"
    }
]

# Fallback to single deployment if additional deployments not configured
available_deployments = []
for deployment in AZURE_DEPLOYMENTS:
    if deployment["deployment"] and deployment["api_key"] and deployment["endpoint"]:
        available_deployments.append(deployment)
    else:
        print(f"Warning: Deployment {deployment['name']} not properly configured, skipping")

if not available_deployments:
    print("Error: No deployments are properly configured!")
    AZURE_DEPLOYMENTS = [AZURE_DEPLOYMENTS[0]]  # Use first as fallback
else:
    AZURE_DEPLOYMENTS = available_deployments

print(f"Configured {len(AZURE_DEPLOYMENTS)} deployment(s): {[d['name'] for d in AZURE_DEPLOYMENTS]}")

# Legacy single deployment support (for backward compatibility)
AZURE_DEPLOYMENT = AZURE_DEPLOYMENTS[0]["deployment"]
AZURE_API_VERSION = AZURE_DEPLOYMENTS[0]["api_version"]

# Model settings
TEMPERATURE = 0.5
MAX_TOKENS = 1500

# Request timeout settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
INACTIVITY_TIMEOUT = 60  # seconds - detect if request is hanging

# System settings
LOG_DIR = "logs"
OUTPUT_DIR = "output"
SIMULATION_ROUNDS = 2

# Create required directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Team configuration
TEAM_NAME = "Multi-Agent Decision Team"
TEAM_GOAL = "Collaborate to solve tasks through structured reasoning and consensus-building"

# Task definition structure - customize for each run
RANKING_TASK = {
    "name": "NASA Lunar Survival",
    "description": """
    You are a member of a space crew originally scheduled to rendezvous with a mother ship on the lighted surface of the moon. 
    Due to mechanical difficulties, however, your ship was forced to land at a spot 200 miles from the rendezvous point. 
    During re-entry and landing, much of the equipment aboard was damaged, and, since survival depends on reaching the mother ship, 
    the most critical items available must be chosen for the 200-mile trip.
    
    Your task is to rank the 15 items below in order of their importance for the crew's survival, starting with 1 = most important.
    """,
    "type": "ranking",  # Options: "ranking", "mcq", "open_ended", "estimation", "selection"
    "options": [
        "Oxygen tanks",
        "Water",
        "Stellar map",
        "Food concentrate",
        "Solar-powered FM receiver-transmitter",
        "50 feet of nylon rope",
        "First aid kit",
        "Parachute silk",
        "Life raft",
        "Signal flares",
        "Two .45 caliber pistols",
        "One case of dehydrated milk",
        "Portable heating unit",
        "Magnetic compass",
        "Box of matches"
    ],
    "expected_output_format": "Ordered list from 1 to 15",
    "ground_truth": [
        "Oxygen tanks",
        "Water",
        "Stellar map",
        "Food concentrate",
        "Solar-powered FM receiver-transmitter",
        "50 feet of nylon rope",
        "First aid kit",
        "Parachute silk",
        "Life raft",
        "Signal flares",
        "Two .45 caliber pistols",
        "One case of dehydrated milk",
        "Portable heating unit",
        "Magnetic compass",
        "Box of matches"
    ],
    "rationale": {
        "Oxygen tanks": "Most pressing survival need on the moon",
        "Water": "Replacement for tremendous liquid loss on the light side",
        "Stellar map": "Primary means of navigation - stars are visible",
        "Food concentrate": "Efficient, high-energy food supply",
        "Solar-powered FM receiver-transmitter": "For communication with mother ship; also possible to use as emergency distress signal",
        "50 feet of nylon rope": "Useful for scaling cliffs, tying injured together",
        "First aid kit": "Valuable for injuries, medications",
        "Parachute silk": "Protection from the sun's rays",
        "Life raft": "CO2 bottles for propulsion across chasms, possible shelter",
        "Signal flares": "Distress call when mother ship is visible",
        "Two .45 caliber pistols": "Possible propulsion devices",
        "One case of dehydrated milk": "Food, mixed with water for drinking",
        "Portable heating unit": "Not needed unless on the dark side",
        "Magnetic compass": "The magnetic field on the moon is not polarized, worthless for navigation",
        "Box of matches": "Virtually worthless - no oxygen to sustain flame"
    }
}

# Example of an MCQ task configuration
MCQ_TASK_EXAMPLE = {
    "name": "Climate Science Question",
    "description": """
    What is the primary greenhouse gas responsible for human-induced climate change?
    """,
    "type": "mcq",
    "options": [
        "A. Carbon dioxide (CO2)",
        "B. Methane (CH4)",
        "C. Water vapor (H2O)",
        "D. Nitrous oxide (N2O)"
    ],
    "expected_output_format": "Single letter selection with rationale",
    "ground_truth": "A",
    "rationale": {
        "A": "While other gases like methane have stronger warming effects per molecule, carbon dioxide is the primary driver of human-induced climate change due to its much larger quantity in the atmosphere and long atmospheric lifetime."
    }
}

# Example of an MCQ task configuration
TASK = {
    "name": "Autoimmune Encephalitis Diagnosis",
    "description": """
    A 32-year-old female presents with subacute onset of memory deficits, confusion, and behavioral changes over 3 weeks. MRI shows bilateral medial temporal lobe hyperintensities. CSF analysis reveals mild lymphocytic pleocytosis. EEG shows focal slowing in the temporal regions. Which autoantibody is most likely associated with this clinical presentation?
    """,
    "type": "mcq",
    "options": [
        "A. Anti-NMDA receptor antibodies",
        "B. Anti-LGI1 antibodies",
        "C. Anti-GABA-B receptor antibodies",
        "D. Anti-AMPA receptor antibodies"
    ],
    "expected_output_format": "Single letter selection with rationale",
    "ground_truth": "B",
    "rationale": {
        "B": "Anti-LGI1 (leucine-rich glioma-inactivated 1) antibodies are strongly associated with limbic encephalitis presenting with subacute memory deficits, confusion, and behavioral changes. The bilateral medial temporal lobe hyperintensities on MRI, mild CSF pleocytosis, and temporal EEG abnormalities are classic findings. While anti-NMDA receptor encephalitis typically presents with more psychiatric symptoms and dyskinesias, anti-GABA-B receptor encephalitis often presents with seizures as the predominant feature, and anti-AMPA receptor encephalitis is less common and often associated with underlying malignancies."
    }
}

# Add after existing TASK definition
TASK_EVALUATION = None  # Stores GT and evaluation data separately from agents

# Example of an open-ended task
OPEN_ENDED_TASK_EXAMPLE = {
    "name": "Business Strategy Question",
    "description": """
    Develop a market entry strategy for a new plant-based meat alternative product in a competitive market dominated by two major brands.
    """,
    "type": "open_ended",
    "expected_output_format": "Structured strategy with key points and rationale",
    "evaluation_criteria": [
        "Identifies target market segments",
        "Addresses competitive positioning",
        "Outlines marketing and distribution approach",
        "Considers pricing strategy",
        "Evaluates risks and mitigations"
    ]
}

# Agent roles and expertise
AGENT_ROLES = {
   "Generalist": "A general medical practitioner with broad knowledge across medical disciplines. "
}

# Decision method weights
DECISION_METHODS = {
    "majority_voting": {
        "name": "Majority Voting",
        "description": "Each agent's preferred answer gets one vote, with the most voted option winning"
    },
    "weighted_voting": {
        "name": "Weighted Voting",
        "description": "Votes are weighted based on agent expertise relevant to the task and confidence levels",
        "weights": {
            "Critical Analyst": 1.0,
            "Domain Expert": 1.0,
            "Creative Strategist": 1.0,
            "Process Facilitator": 1.0
        }
    },
    "borda_count": {
        "name": "Borda Count",
        "description": "Each agent ranks all options, assigning points based on position (n-k points for k-th place in a list of n options)"
    }
}

# Agent recruitment settings
RECRUITMENT_METHOD = "adaptive"  # Options: "adaptive", "fixed", "basic", "intermediate", "advanced"
RECRUITMENT_POOLS = {
    "medical": [
        "Cardiologist - Specializes in the heart and cardiovascular system",
        "Neurologist - Focuses on the brain and nervous system disorders",
        "Pulmonologist - Specializes in respiratory system and lung diseases",
        "Endocrinologist - Focuses on hormonal systems and metabolic disorders",
        "Gastroenterologist - Specializes in digestive system disorders",
        "Oncologist - Focuses on cancer diagnosis and treatment",
        "Pediatrician - Specializes in child and adolescent health",
        "Psychiatrist - Focuses on mental health disorders",
        "Rheumatologist - Specializes in autoimmune and joint disorders",
        "Hematologist - Focuses on blood disorders and diseases",
        "Medical Geneticist - Specializes in the study of genes and heredity",
        "Neonatologist - Focuses on the care of newborn infants, especially those who are premature or have medical issues",
        "Otolaryngologist - Specializes in ear, nose, and throat disorders (ENT Surgeon)",
        "General Surgeon - Provides surgical expertise for a wide range of conditions",
        "Anesthesiologist - Focuses on perioperative care and pain management",
        "Speech-Language Pathologist - Specializes in communication and swallowing disorders",
        "Physical Therapist - Offers rehabilitation services to improve movement and manage pain",
        "Vocational Therapist - Assists patients in adapting to health changes affecting their occupation",
        "Clinical Decision Specialist - Coordinates recommendations and formulates comprehensive treatment plans",
    ],

    "general": [
        "Medical Generalist - A general medical practitioner with broad knowledge across medical disciplines."
    ]
}


# Enhanced recruitment prompt for MedMCQA
RECRUITMENT_PROMPTS_MEDMCQA = {
    "complexity_evaluation_medmcqa": """
    Analyze the following medical question and its options to determine complexity level:
    
    Question: {question}
    
    Answer Options:
    {options}
    
    Subject: {subject}
    Topic: {topic}
    
    Consider these factors:
    - Subject area complexity (basic sciences vs. specialized clinical)
    - Topic specificity and depth required
    - Whether options indicate multi-system involvement
    - Diagnostic vs. therapeutic vs. mechanism-based question
    - Whether options suggest rare conditions or common presentations
    
    Based on analysis, classify as:
    1) basic - Single domain knowledge, common conditions/concepts
    2) intermediate - Multi-domain or specialized knowledge required  
    3) advanced - Complex differential diagnosis, rare conditions, or multi-system integration
    
    **Complexity Classification:** [number]) [complexity level]
    """,
    
    "team_selection_medmcqa": """You are recruiting medical experts to answer this question collaboratively.
        
    Question: {question}

    Answer Options:
    {options}
    
    Subject Area: {subject}
    Clinical Topic: {topic}

    Based on the question content, answer options, and subject area, recruit {num_agents} experts with DISTINCT specialties that are directly relevant.

    Consider the options carefully - they often reveal the specific medical domains needed:
    - Anatomical structures mentioned → relevant specialists
    - Disease processes → corresponding subspecialists  
    - Diagnostic findings → interpreting specialists
    - Treatment modalities → managing specialists
    - Physiological mechanisms → basic science experts

    For each expert, assign weight (0.0-1.0) reflecting importance to this specific question. Total weights should sum to 1.0.

    Specify communication structure or mark as Independent.

    Example format:
    1. [Specialist] - [Expertise description] - Hierarchy: [Structure] - Weight: [0.0-1.0]
    2. [Specialist] - [Expertise description] - Hierarchy: [Structure] - Weight: [0.0-1.0]
    ...

    Do not include reasoning, just the recruitment list.
    """,
    
    "mdt_design_medmcqa": """You are organizing Multidisciplinary Teams (MDTs) for this complex medical question.

Question: {question}

Answer Options:
{options}

Subject: {subject}
Topic: {topic}

The answer options suggest multiple domains of expertise are needed. Organize 3 MDTs with 3 clinicians each.

Consider how the options inform team composition:
- Options mentioning specific organ systems → relevant specialists
- Diagnostic options → imaging/lab specialists  
- Treatment options → therapeutic specialists
- Mechanism options → basic science experts

Include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT).

Format:
Group 1 - Initial Assessment Team (IAT)
Member 1: [Specialist] (Lead) - [Expertise and relevance to options]
Member 2: [Specialist] - [Expertise and relevance to options]  
Member 3: [Specialist] - [Expertise and relevance to options]

Group 2 - [Domain-specific Team Name]
Member 1: [Specialist] (Lead) - [Expertise and relevance to options]
Member 2: [Specialist] - [Expertise and relevance to options]
Member 3: [Specialist] - [Expertise and relevance to options]

Group 3 - Final Review and Decision Team (FRDT)
Member 1: [Specialist] (Lead) - [Expertise and relevance to options]
Member 2: [Specialist] - [Expertise and relevance to options]
Member 3: [Specialist] - [Expertise and relevance to options]
"""
}


# Configure which teamwork components to use
USE_TEAM_LEADERSHIP = False
USE_CLOSED_LOOP_COMM = False
USE_MUTUAL_MONITORING = False
USE_SHARED_MENTAL_MODEL = False
USE_TEAM_ORIENTATION = False  # Default disabled
USE_MUTUAL_TRUST = False  # Default disabled

MUTUAL_TRUST_FACTOR = 0.9  # Default trust level (0.0-1.0)

USE_AGENT_RECRUITMENT = False  # Default disabled

# Parallel processing settings - Updated for question-level parallelism
ENABLE_QUESTION_PARALLEL = len(AZURE_DEPLOYMENTS) > 1  # Enable question-level parallel processing
MAX_PARALLEL_QUESTIONS = len(AZURE_DEPLOYMENTS)  # Maximum questions to process in parallel

# Legacy parallel processing settings (deprecated but kept for compatibility)
ENABLE_PARALLEL_PROCESSING = False  # Agent-level parallelism is now disabled
MAX_PARALLEL_WORKERS = 1  # Always use sequential agent processing

def get_deployment_for_agent(agent_index: int) -> Dict[str, str]:
    """
    Get deployment configuration for agent based on round-robin distribution.
    
    NOTE: This function is now used less frequently due to the switch to 
    question-level parallel processing, but kept for compatibility.
    
    Args:
        agent_index: Index of the agent (0-based)
        
    Returns:
        Deployment configuration dictionary
    """
    deployment_index = agent_index % len(AZURE_DEPLOYMENTS)
    return AZURE_DEPLOYMENTS[deployment_index]

def get_deployment_for_question(question_index: int) -> Dict[str, str]:
    """
    Get deployment configuration for question based on round-robin distribution.
    
    This is the primary method for assigning deployments in question-level parallelism.
    
    Args:
        question_index: Index of the question (0-based)
        
    Returns:
        Deployment configuration dictionary
    """
    deployment_index = question_index % len(AZURE_DEPLOYMENTS)
    return AZURE_DEPLOYMENTS[deployment_index]

def get_all_deployments() -> List[Dict[str, str]]:
    """Get all available deployment configurations."""
    return AZURE_DEPLOYMENTS.copy()

def get_parallel_processing_info() -> Dict[str, Any]:
    """
    Get information about the current parallel processing configuration.
    
    Returns:
        Dictionary with parallel processing details
    """
    return {
        "question_level_parallel": ENABLE_QUESTION_PARALLEL,
        "max_parallel_questions": MAX_PARALLEL_QUESTIONS,
        "num_deployments": len(AZURE_DEPLOYMENTS),
        "deployment_names": [d['name'] for d in AZURE_DEPLOYMENTS],
        "agent_level_parallel": ENABLE_PARALLEL_PROCESSING,  # Should be False
        "processing_mode": "question_level" if ENABLE_QUESTION_PARALLEL else "sequential"
    }

# Print configuration info on import
if __name__ == "__main__":
    print("\n=== Agent System Configuration ===")
    parallel_info = get_parallel_processing_info()
    print(f"Processing Mode: {parallel_info['processing_mode']}")
    print(f"Available Deployments: {parallel_info['num_deployments']}")
    print(f"Deployment Names: {parallel_info['deployment_names']}")
    
    if parallel_info['question_level_parallel']:
        print(f"Max Parallel Questions: {parallel_info['max_parallel_questions']}")
        print("Note: Questions will be distributed across deployments in round-robin fashion")
    else:
        print("Sequential processing mode (single deployment)")
    
    print("=====================================\n")