# Big5-Agents

A flexible multi-agent system with modular teamwork components based on the Big Five teamwork model. Agents collaborate on various tasks with configurable teamwork behaviors.

## Table of Contents
- [Overview](#overview)
- [Big Five Teamwork Model](#big-five-teamwork-model)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command Line Options](#command-line-options)
  - [Running Multiple Configurations](#running-multiple-configurations)
- [Dataset Runner](#dataset-runner)
  - [Dataset Runner Options](#dataset-runner-options)
  - [Examples](#dataset-runner-examples)
- [Agent Recruitment](#agent-recruitment)
  - [Complexity Levels](#complexity-levels)
  - [Recruitment Methods](#recruitment-methods)
  - [Recruitment Pools](#recruitment-pools)
- [Configuration](#configuration)
  - [Task Configuration](#task-configuration)
  - [Agent Roles & Recruitment Pools](#agent-roles--recruitment-pools)
  - [Decision Methods](#decision-methods)
- [Logs Structure](#logs-structure)
- [Results Output](#results-output)
- [Project Structure](#project-structure)
- [Extending the System](#extending-the-system)
- [References](#references)

## Overview

This system implements a multi-agent approach for collaborative problem solving with the following features:

- Dynamic task handling (ranking, MCQ, open-ended questions)
- Modular teamwork components that can be toggled independently
- Specialized agent roles with distinct expertise
- Multiple decision aggregation methods
- Comprehensive logging system with separate channels
- Performance evaluation against ground truth
- **Adaptive agent recruitment** based on task complexity
- **Dataset processing** for batch evaluation

The system is built around the Big Five teamwork model components.

## Big Five Teamwork Model

Our system is built on the Big Five teamwork model introduced by Salas et al. (2005), which identifies five core components of effective teamwork:

1. **Team Leadership** - Coordinating activities and defining approaches
2. **Mutual Performance Monitoring** - Tracking teammates' work and providing feedback
3. **Backup Behavior** - Providing support via specialized knowledge
4. **Adaptability** - Adjusting strategies based on information exchange
5. **Team Orientation** - Shared mental models for better coordination

These components are augmented by three coordinating mechanisms:
- Shared Mental Models
- Closed-Loop Communication
- Mutual Trust

## Installation

```bash
# Clone the repository
git clone https://github.com/V-ARE/mafia-via-agents/tree/main/Big5-Agents.git
cd big5-agents

# Install dependencies
pip install -r requirements.txt

# Set up your API keys in .env file
cp .env.example .env
```

## Usage

### Basic Usage

```bash
# Run simulation with all teamwork components
python main.py

# Run simulation with specific components
python main.py --leadership --closedloop --mutual --mental --orientation

# Run with mutual trust and custom trust factor
python main.py --trust --trust-factor 0.6

# Run with dynamic agent recruitment
python main.py --recruitment
```

### Command Line Options

```
--leadership      Enable team leadership component
--closedloop      Enable closed-loop communication
--mutual          Enable mutual performance monitoring
--mental          Enable shared mental model
--orientation     Enable Team Orientataion
--trust           Enable Mutual Trust
--trust-factor    Custom trust factor[0-1]
--all             Run all feature combinations
--random-leader   Randomly assign leadership
--runs N          Number of runs for each configuration (default: 1)
--recruitment     Enable dynamic agent recruitment
--recruitment-method {adaptive|basic|intermediate|advanced}
--recruitment-pool {general|medical}
```

### Running Multiple Configurations

```bash
# Run all possible combinations with 3 runs each
python main.py --all --runs 3
```

## Dataset Runner

The dataset runner allows you to evaluate your agent system on datasets like MedQA and PubMedQA:

### Dataset Runner Options

```
--dataset TYPE         Dataset to run (medqa or pubmedqa)
--num-questions N      Number of questions to process (default: 50)
--seed N               Random seed for reproducibility (default: 42)
--all                  Run all feature configurations
--output-dir PATH      Output directory for results
--leadership           Use team leadership
--closedloop           Use closed-loop communication
--mutual               Use mutual performance monitoring
--mental               Use shared mental model
--orientation          Use team orientation
--trust                Use mutual trust
--trust-factor N       Mutual trust factor (0.0-1.0) (default: 0.8)
--recruitment          Use dynamic agent recruitment
--recruitment-method   Method for recruitment (adaptive, basic, intermediate, advanced)
--recruitment-pool     Pool of agent roles to recruit from (general, medical)
```

### Dataset Runner Examples

```bash
# Run 50 random MedQA questions with all teamwork components
python dataset_runner.py --dataset medqa --num-questions 50 --leadership --closedloop --mutual --mental

# Run with recruitment of medical specialists
python dataset_runner.py --dataset medqa --recruitment --recruitment-pool medical

# Run all configurations (baseline, individual components, all components) on PubMedQA
python dataset_runner.py --dataset pubmedqa --num-questions 25 --all

# Specify custom output directory and random seed
python dataset_runner.py --dataset medqa --output-dir ./results --seed 123 --all
```

## Agent Recruitment

The system supports dynamic agent team assembly based on task complexity:

### Complexity Levels

- **Basic**: Single expert for straightforward questions
- **Intermediate**: Team of specialists (5 by default) with hierarchical relationships
- **Advanced**: Multiple specialized teams (3 teams of 3 experts) with a chief coordinator

### Recruitment Methods

- **Adaptive**: Automatically determines complexity level based on question analysis
- **Fixed**: Use pre-determined complexity level (basic, intermediate, advanced)

### Recruitment Pools

- **General**: Generic roles suitable for diverse tasks (Critical Analyst, Domain Expert, etc.)
- **Medical**: Specialized medical roles for healthcare questions (Cardiologist, Neurologist, etc.)

## Configuration

The `config.py` file contains all configuration settings:

### Task Configuration

Edit the `TASK` dictionary to change the task:

```python
TASK = {
    "name": "NASA Lunar Survival",
    "description": "Rank items in order of importance for a lunar trek...",
    "type": "ranking",  # Options: "ranking", "mcq", "open_ended"
    "options": [...],  # List of items or options
    "expected_output_format": "Ordered list from 1 to 15",
    "ground_truth": [...],  # Optional ground truth for evaluation
    "rationale": {...}  # Optional reasoning for ground truth
}
```

### Prompt Management

The `utils/prompts.py` file centralizes all prompts used throughout the system, making it easy to modify agent behaviors without changing the core code.

**Structure**
Prompts are organized into categorical dictionaries:

```python
# Agent system prompts
AGENT_SYSTEM_PROMPTS = {
    "base": "Base system prompt template...",
    # Additional specialized prompts...
}

# Team leadership prompts
LEADERSHIP_PROMPTS = {
    "team_leadership": "Leadership component prompt...",
    "define_task": "Task definition prompt...",
    # Other leadership actions...
}

# Additional categories (COMMUNICATION_PROMPTS, MONITORING_PROMPTS, etc.)

```

**Modifying Prompts**
To modify agent behavior for specific components:

- Open utils/prompts.py
- Locate the appropriate prompt category (e.g., TRUST_PROMPTS)
- Edit the prompt text while preserving format placeholders (e.g., {role}, {task_description})
- Save the file - no other code changes required

### Agent Roles & Recruitment Pools

Define agent roles and recruitment pools:

```python
AGENT_ROLES = {
    "Critical Analyst": "Approaches problems with analytical rigor...",
    "Domain Expert": "Provides specialized knowledge...",
    # Add/modify roles as needed
}

RECRUITMENT_POOLS = {
    "medical": [
        "Cardiologist - Specializes in the heart and cardiovascular system",
        "Neurologist - Focuses on the brain and nervous system disorders",
        # More specialists...
    ],
    "general": [
        "Critical Analyst - Approaches problems with analytical rigor...",
        "Domain Expert - Provides specialized knowledge...",
        # More general roles...
    ]
}
```

### Decision Methods

Configure decision method weights:

```python
DECISION_METHODS = {
    "weighted_voting": {
        "weights": {
            "Critical Analyst": 1.2,
            "Domain Expert": 1.5,
            # Adjust weights
        }
    }
}
```

## Logs Structure

The system generates detailed logs in the `logs/` directory with the following structure:

```
logs/
├── baseline/                            # No teamwork components
│   └── sim_20250417_123456/
├── leadership/                          # Only leadership enabled
├── recruitment_basic/                   # Basic recruitment
├── recruitment_intermediate/            # Intermediate recruitment
├── leadership_closed_loop_mutual_monitoring_shared_mental_model/  # All components
└── leadership_closed_loop_mutual_monitoring_shared_mental_model_recruitment/  # All with recruitment
```

## Results Output

Simulation results are saved to the `output/` directory:

```
output/
├── sim_20250417_123456_results.json   # Individual simulation results
├── all_configurations_results.json    # Aggregated results from --all runs
├── medqa_results/                     # Dataset runner results
│   ├── baseline/
│   ├── recruitment_adaptive/
│   └── combined_results.json
```

## Project Structure

```
big5_agents/
├── components/                     # Big Five components 
│   ├── agent.py                    # Base agent class
│   ├── modular_agent.py            # Specialized agent implementation
│   ├── agent_recruitment.py        # Dynamic agent team assembly
│   ├── closed_loop.py              # Closed-loop communication
│   ├── mutual_monitoring.py        # Performance monitoring
│   ├── shared_mental_model.py      # Shared understanding management
│   ├── decision_methods.py         # Decision aggregation methods
│   ├── team_orientation.py         # Team orientation component
│   └── mutual_trust.py             # Mutual trust component
├── utils/
│   ├── logger.py                   # Enhanced logging system
│   └── prompts.py                  # Centralized prompt management
├── simulator.py                    # Main simulation orchestration
├── config.py                       # Configuration settings
├── main.py                         # Command-line interface
├── dataset_runner.py               # Dataset processing utility
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Extending the System

- Add new agent roles in `config.AGENT_ROLES` or `config.RECRUITMENT_POOLS`
- Implement new task types in `agent.extract_response()`
- Create additional decision methods in `decision_methods.py`
- Add new recruitment strategies in `agent_recruitment.py`
- Modify prompts centrally in `utils/prompts.py`


## References

- Kim, Y., Park, C., Jeong, H., Chan, Y. S., Xu, X., McDuff, D., Lee, H., Ghassemi, M., Breazeal, C., & Park, H. W. (2024). MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making. arXiv preprint arXiv:2404.15155. https://arxiv.org/abs/2404.15155
- Salas, E., Sims, D. E., & Burke, C. S. (2005). Is there a "Big Five" in Teamwork?. Small Group Research, 36(5), 555-599. https://doi.org/10.1177/1046496405277134