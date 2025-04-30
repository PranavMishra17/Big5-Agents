"""
Configuration settings for the modular agent system.
"""
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Azure OpenAI settings
AZURE_API_KEY = "428KgVArXb6sFyseVYDjElDDYZnlCnx8pNa8CfU5dCic6gjOK89WJQQJ99BBACYeBjFXJ3w3AAABACOG5gtQ" #os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT')
AZURE_DEPLOYMENT = "VARELab-GPT4o"
AZURE_API_VERSION = "2024-08-01-preview"

# Model settings
TEMPERATURE = 0.5
MAX_TOKENS = 1500

# System settings
LOG_DIR = "logs"
OUTPUT_DIR = "output"
SIMULATION_ROUNDS = 3

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
    "Critical Analyst": "Approaches problems with analytical rigor, questioning assumptions and evaluating evidence. Brings expertise in logical reasoning, statistical analysis, and critical thinking methodology.",
    
    "Domain Expert": "Provides specialized knowledge relevant to the task domain. Has deep understanding of principles, historical context, and technical aspects of the subject matter.",
    
    "Creative Strategist": "Offers innovative perspectives and approaches to problem solving. Specializes in making connections between disparate concepts and thinking beyond conventional solutions.",
    
    "Process Facilitator": "Focuses on optimizing the collaborative process and ensuring methodical evaluation. Brings expertise in decision frameworks, consensus building, and structured problem solving."
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
        "Hematologist - Focuses on blood disorders and diseases"
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
        "Critical Analyst - Approaches problems with analytical rigor, questioning assumptions and evaluating evidence",
        "Domain Expert - Provides specialized knowledge relevant to the task domain",
        "Creative Strategist - Offers innovative perspectives and approaches to problem solving",
        "Process Facilitator - Focuses on optimizing the collaborative process and ensuring methodical evaluation",
        "Systems Thinker - Analyzes how different components interact and affect each other",
        "Data Specialist - Evaluates quantitative information and statistical patterns",
        "Risk Assessor - Identifies potential problems and evaluates their likelihood and impact",
        "Implementation Expert - Focuses on practical execution and operational constraints"
    ]
}

# Configure which teamwork components to use
USE_TEAM_LEADERSHIP = True
USE_CLOSED_LOOP_COMM = True
USE_MUTUAL_MONITORING = True
USE_SHARED_MENTAL_MODEL = True


USE_AGENT_RECRUITMENT = False  # Default disabled