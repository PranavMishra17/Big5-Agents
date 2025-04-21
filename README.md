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

The system is built around the Big Five teamwork model components:
1. **Team Leadership** - Coordinating activities and defining approaches
2. **Mutual Performance Monitoring** - Tracking teammates' work and providing feedback
3. **Backup Behavior** - Providing support via specialized knowledge
4. **Adaptability** - Adjusting strategies based on information exchange
5. **Team Orientation** - Shared mental models for better coordination

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/modular-agent-system.git
cd modular-agent-system

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
python main.py --leadership --closedloop --mutual --mental
```

### Command Line Options

```
--leadership      Enable team leadership component
--closedloop      Enable closed-loop communication
--mutual          Enable mutual performance monitoring
--mental          Enable shared mental model
--all             Run all feature combinations
--random-leader   Randomly assign leadership
--runs N          Number of runs for each configuration (default: 1)
```

### Running Multiple Configurations

```bash
# Run all possible combinations with 3 runs each
python main.py --all --runs 3
```

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

### Agent Roles

Define agent roles and their expertise:

```python
AGENT_ROLES = {
    "Critical Analyst": "Approaches problems with analytical rigor...",
    "Domain Expert": "Provides specialized knowledge...",
    # Add/modify roles as needed
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
│       ├── sim_20250417_123456.log      # Main log file
│       ├── sim_20250417_123456_events.jsonl           # General events
│       ├── sim_20250417_123456_main_discussion.jsonl  # Primary agent interactions
├── leadership/                          # Only leadership enabled
├── leadership_closed_loop/              # Two components enabled
├── leadership_closed_loop_mutual_monitoring_shared_mental_model/  # All components
    └── sim_20250417_789012/
        ├── sim_20250417_789012.log                    # Main log file
        ├── sim_20250417_789012_events.jsonl           # General events
        ├── sim_20250417_789012_main_discussion.jsonl  # Primary agent interactions
        ├── sim_20250417_789012_closed_loop.jsonl      # Closed-loop communications
        ├── sim_20250417_789012_leadership.jsonl       # Leadership actions
        ├── sim_20250417_789012_monitoring.jsonl       # Monitoring actions
        ├── sim_20250417_789012_mental_model.jsonl     # Mental model updates
        ├── sim_20250417_789012_decision.jsonl         # Decision method outputs
```

### Log Channels Explained

1. **Main Discussion** - Core dialogue between agents
2. **Closed-Loop** - Three-step communication exchanges (message, acknowledgment, verification)
3. **Leadership** - Leader-specific actions (task definition, synthesis)
4. **Monitoring** - Feedback and issue detection between agents
5. **Mental Model** - Shared understanding and convergence tracking
6. **Decision** - Results from different decision aggregation methods

## Results Output

Simulation results are saved to the `output/` directory:

```
output/
├── sim_20250417_123456_results.json   # Individual simulation results
├── all_configurations_results.json    # Aggregated results from --all runs
```

### Evaluation Metrics

For ranking tasks:
- **Correlation**: Spearman's rank correlation with ground truth (-1 to 1)
- **Error**: Sum of absolute position differences (lower is better)

For MCQ tasks:
- **Accuracy**: Ratio of correct answers
- **Confidence**: Agent's confidence in the answer

## Decision Methods

The system implements three decision methods:

1. **Majority Voting** - Each agent gets one vote of equal weight
2. **Weighted Voting** - Votes are weighted by agent expertise and confidence
3. **Borda Count** - Points assigned based on preference ranking

Each method is applied independently, allowing comparison of results.

## Project Structure

```
modular_agent_system/
├── agent.py                 # Base agent class
├── modular_agent.py         # Specialized agent implementation
├── closed_loop.py           # Closed-loop communication
├── mutual_monitoring.py     # Performance monitoring
├── shared_mental_model.py   # Shared understanding management
├── decision_methods.py      # Decision aggregation methods
├── simulator.py             # Main simulation orchestration
├── logger.py                # Enhanced logging system
├── main.py                  # Command-line interface
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Example Output

When running a simulation, you'll see output like:

```
Task: NASA Lunar Survival
Type: ranking

Results Summary:
Teamwork Features: Team Leadership, Closed-loop Communication, Mutual Performance Monitoring
  
Decision Method Results:

Majority Voting:
  Top 3 items:
    1. Oxygen tanks
    2. Water
    3. Stellar map
  Confidence: 0.83

Weighted Voting:
  Top 3 items:
    1. Oxygen tanks
    2. Water
    3. Stellar map
  Confidence: 0.91

Borda Count:
  Top 3 items:
    1. Oxygen tanks
    2. Water
    3. Stellar map
  Confidence: 0.87

Performance Summary:
  Majority Voting:
    Correlation: 0.9524
    Error: 4
  Weighted Voting:
    Correlation: 0.9762
    Error: 2
  Borda Count:
    Correlation: 0.9643
    Error: 3

Detailed logs available at: logs/leadership_closed_loop_mutual_monitoring/sim_20250417_123456
```

## Extending the System

- Add new agent roles in `config.AGENT_ROLES`
- Implement new task types in `agent.extract_response()`
- Create additional decision methods in `decision_methods.py`


## How to Use the Dataset Runner
```bash
# Run 50 random MedQA questions with all teamwork components
python dataset_runner.py --dataset medqa --num-questions 50 --leadership --closedloop --mutual --mental

# Run all configurations (baseline, individual components, all components) on PubMedQA
python dataset_runner.py --dataset pubmedqa --num-questions 25 --all

# Specify custom output directory and random seed
python dataset_runner.py --dataset medqa --output-dir ./results --seed 123 --all
```
## Key Features

Flexible Dataset Support: The system automatically formats different dataset structures into the appropriate task format for your agent system.
Modular Configuration: You can run with any combination of the Big Five teamwork components.
Results Analysis: For each configuration, you get accuracy metrics for all three decision methods.
Structured Output: Results are saved in a hierarchical format with summary statistics and detailed per-question results.
Progress Tracking: Uses tqdm to show progress during long runs with many questions.

The dataset runner integrates smoothly with your existing agent system by directly configuring the config.TASK for each question. The results are stored in a structured format that facilitates further analysis and comparison.
If you want to add more datasets in the future, you would just need to:

### Add a new load_X_dataset function
Create a new format_X_for_task function to convert to your standard task format
Add the dataset type to the command-line choices