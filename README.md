# Big5-Agents

## Modular Agent System

A flexible multi-agent system with modular teamwork components based on the Big Five teamwork model. Agents collaborate on various tasks with configurable teamwork behaviors.

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

The system is built around the Big Five teamwork model components:
1. **Team Leadership** - Coordinating activities and defining approaches
2. **Mutual Performance Monitoring** - Tracking teammates' work and providing feedback
3. **Backup Behavior** - Providing support via specialized knowledge
4. **Adaptability** - Adjusting strategies based on information exchange
5. **Team Orientation** - Shared mental models for better coordination

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/big5-agents.git
cd big5-agents

# Install dependencies
pip install -r requirements.txt

# Set up your API keys in .env file
cp .env.example .env
# Edit .env with your API keys
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
├── agent.py                 # Base agent class
├── modular_agent.py         # Specialized agent implementation
├── agent_recruitment.py     # Dynamic agent team assembly
├── closed_loop.py           # Closed-loop communication
├── mutual_monitoring.py     # Performance monitoring
├── shared_mental_model.py   # Shared understanding management
├── decision_methods.py      # Decision aggregation methods
├── simulator.py             # Main simulation orchestration
├── logger.py                # Enhanced logging system
├── main.py                  # Command-line interface
├── dataset_runner.py        # Dataset processing utility
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Extending the System

- Add new agent roles in `config.AGENT_ROLES` or `config.RECRUITMENT_POOLS`
- Implement new task types in `agent.extract_response()`
- Create additional decision methods in `decision_methods.py`
- Add new recruitment strategies in `agent_recruitment.py`