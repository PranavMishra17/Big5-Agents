# Big5-Agents [Submitted to AAAI as TeamMedAgents]

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![arXiv](https://img.shields.io/badge/arXiv-2508.08115-red.svg)
![Status](https://img.shields.io/badge/Status-Under%20Review-yellow.svg)

![Framework](https://github.com/PranavMishra17/Big5-Agents/blob/186d3f604b5597ce5f5f6ebfb020279e55b742c3/frame.png)

> *A flexible multi-agent system with modular teamwork components based on the Big Five teamwork model. Agents collaborate on various tasks with configurable teamwork behaviors.*

📄 **Paper:** [TeamMedAgents: Enhancing Medical Decision-Making Through Structured Teamwork](https://arxiv.org/abs/2508.08115)

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
6. **Shared Mental Models**

7. **Closed-Loop Communication**

8. **Mutual Trust**


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

## 🆕 Dynamic Features

### Dynamic Teamwork Configuration
- **Automatic Component Selection**: AI automatically selects up to 3 teamwork components based on question analysis
- **Question-Adaptive**: Each question gets analyzed to determine optimal collaboration strategy
- **Component Options**: Leadership, Closed-Loop Communication, Mutual Monitoring, Shared Mental Model, Team Orientation, Mutual Trust

### Dynamic Agent Recruitment
- **Adaptive Team Size**: Automatically determines optimal team size (2-5 agents) per question
- **Complexity-Based**: Team size and composition adapt to question complexity and scope
- **Efficiency Optimized**: Balances expertise diversity with coordination overhead


### Basic Usage (Dynamic Configuration)

```bash
# Run with dynamic selection (default)
python dataset_runner.py --dataset medqa --num-questions 20

# Run with dynamic selection explicitly enabled
python dataset_runner.py --dataset medmcqa --num-questions 50 --enable-dynamic-selection
```

### Static Configuration (Backward Compatibility)

```bash
# Disable dynamic selection and use static config
python dataset_runner.py --dataset pubmedqa --num-questions 30 --disable-dynamic-selection --leadership --closedloop

# Specific team size (disables dynamic sizing)
python dataset_runner.py --dataset ddxplus --num-questions 25 --n-max 4 --leadership --mutual
```

### Advanced Usage

```bash
# Run all configurations including dynamic
python dataset_runner.py --dataset medbullets --num-questions 40 --all

# Custom recruitment with dynamic selection
python dataset_runner.py --dataset medqa --recruitment --recruitment-method adaptive --recruitment-pool medical

# Vision-enabled datasets with dynamic configuration
python dataset_runner.py --dataset pmc_vqa --num-questions 20 --enable-dynamic-selection
```

## 📊 Supported Datasets

### Text-Only Datasets
- **MedQA**: Medical multiple-choice questions with explanations
- **MedMCQA**: Medical entrance exam questions  
- **PubMedQA**: Yes/No/Maybe research questions with abstracts
- **MMLU-Pro Medical**: Advanced medical knowledge questions
- **DDXPlus**: Clinical diagnosis cases with patient symptoms
- **MedBullets**: USMLE-style clinical questions

### Vision-Enabled Datasets  
- **PMC-VQA**: Medical image questions from research papers
- **Path-VQA**: Pathology slide analysis questions

## 🔧 Configuration Options

### Dynamic Selection Parameters

```bash
--enable-dynamic-selection    # Enable AI-driven team configuration (default: True)
--disable-dynamic-selection   # Use static configuration only
```

### Teamwork Components (Static Mode)

```bash
--leadership              # Enable team leadership
--closedloop             # Enable closed-loop communication  
--mutual                 # Enable mutual monitoring
--mental                 # Enable shared mental model
--orientation            # Enable team orientation
--trust                  # Enable mutual trust
--trust-factor 0.8       # Set mutual trust factor (0.0-1.0)
```

### Agent Recruitment

```bash
--recruitment                    # Enable dynamic agent recruitment
--recruitment-method adaptive    # Method: adaptive, basic, intermediate, advanced
--recruitment-pool medical       # Pool: general, medical
--n-max 5                       # Maximum agents (disables dynamic sizing if set)
```

### Dataset and Processing

```bash
--dataset medqa              # Dataset to use
--num-questions 50          # Number of questions to process
--seed 42                   # Random seed for reproducibility
--all                       # Run all configurations for comparison
--output-dir ./results      # Output directory for results
```

## 📈 Understanding Results

### Basic Output
```
Summary for Dynamic Configuration on medqa:
  Majority Voting: 42/50 correct (84.00%)
  Weighted Voting: 45/50 correct (90.00%)
  Borda Count: 43/50 correct (86.00%)
  
Dynamic Selection:
  Team sizes used: {'3': 15, '4': 20, '5': 15}
  Teamwork configs used: {'leadership,monitoring': 25, 'leadership,closedloop,trust': 25}
```

### Detailed Analysis Files
- `summary.json`: Performance metrics by decision method
- `dynamic_selection_results.json`: Dynamic selection patterns and statistics
- `disagreement_analysis.json`: Agent disagreement patterns  
- `complexity_distribution.json`: Question complexity analysis
- `detailed_results_enhanced.json`: Complete simulation data

## 🔍 Advanced Features

### Vision Processing
```bash
# Automatic vision agent recruitment for image datasets
python dataset_runner.py --dataset pmc_vqa --enable-dynamic-selection

# Pathology-specific processing
python dataset_runner.py --dataset path_vqa --recruitment-pool medical
```

### Multi-Deployment Support
```bash
# Parallel processing across multiple deployments
# Automatically distributes questions across available deployments
# Check deployment_usage.json for load distribution
```

### Error Recovery and Validation
```bash
# Automatic validation error handling
# Vision fallback for corrupted images  
# Retry mechanisms with exponential backoff
```


### Explicit Configuration (Disables Dynamic Selection)
```bash
# Any explicit teamwork parameter disables dynamic selection
python dataset_runner.py --leadership --closedloop  # Static config used

# Explicit team size disables dynamic sizing
python dataset_runner.py --n-max 4  # Uses exactly 4 agents

# Combination disables all dynamic features
python dataset_runner.py --n-max 3 --leadership --mutual  # Fully static
```

### Legacy API Support
```python
# All existing function calls continue to work
from simulator import AgentSystemSimulator

# Old way (still works)
sim = AgentSystemSimulator(
    use_team_leadership=True,
    use_closed_loop_comm=True,
    n_max=4
)

# New way (with dynamic selection)
sim = AgentSystemSimulator(
    enable_dynamic_selection=True  # AI determines optimal config
)
```

## 📊 Performance Analysis

### Dynamic vs Static Comparison
```bash
# Compare dynamic vs static configurations
python dataset_runner.py --dataset medqa --all --num-questions 100

# Results show performance across:
# - Dynamic Configuration (adapts per question)
# - Static Leadership (always uses leadership)
# - Static Closed-Loop (always uses closed-loop)
# - Static All Features (uses all components)
```

### Configuration Effectiveness
```python
# Analyze which configurations work best for different question types
{
  "Dynamic Configuration": {
    "accuracy": 0.87,
    "team_sizes_used": {"3": 15, "4": 20, "5": 15},
    "components_used": {"leadership,monitoring": 25, "leadership,trust": 25}
  },
  "Static All Features": {
    "accuracy": 0.82,
    "overhead": "High coordination overhead for simple questions"
  }
}
```

## 🛠️ Development and Customization

### Adding New Teamwork Components
```python
# 1. Create component in components/ directory
# 2. Add to DYNAMIC_RECRUITMENT_PROMPTS in utils/prompts.py
# 3. Update component_mapping in determine_optimal_teamwork_config()
# 4. Add initialization in AgentSystemSimulator
```

### Custom Dynamic Selection Logic
```python
# Modify determine_optimal_teamwork_config() in agent_recruitment.py
def determine_optimal_teamwork_config(question, complexity, team_size):
    # Your custom logic here
    # Return dict with teamwork component selections
    pass
```

### Adding New Datasets
```python
# 1. Create load_your_dataset() function in dataset_runner.py
# 2. Create format_your_dataset_for_task() function
# 3. Add dataset choice to main() argument parser
# 4. Add vision support if needed
```

## 🧪 Testing and Validation

### Quick Test
```bash
# Test with small dataset
python dataset_runner.py --dataset medqa --num-questions 5 --enable-dynamic-selection

# Validate all datasets
python dataset_runner.py --validate-only
```

### Configuration Testing
```bash
# Test static vs dynamic on same questions
python dataset_runner.py --dataset medmcqa --num-questions 20 --seed 42 --enable-dynamic-selection
python dataset_runner.py --dataset medmcqa --num-questions 20 --seed 42 --disable-dynamic-selection --leadership
```

### Error Simulation
```bash
# Test error recovery with challenging datasets
python dataset_runner.py --dataset pmc_vqa --num-questions 10  # Vision errors
python dataset_runner.py --dataset medmcqa --num-questions 50  # Validation errors
```

## 📝 Logging and Monitoring

### Enhanced Logging
```python
# Dynamic selection decisions are logged
2024-01-01 10:00:00 [INFO] Dynamic team size selection: 4 agents
2024-01-01 10:00:01 [INFO] Dynamic teamwork configuration: ['leadership', 'monitoring']
2024-01-01 10:00:02 [INFO] Selection rationale: Complex diagnostic case requiring coordination

# Vision processing logs
2024-01-01 10:00:03 [INFO] Vision task detected - using specialized vision-capable recruitment
2024-01-01 10:00:04 [INFO] Pathology Specialist: Using specialized pathology analysis
```

### Monitoring Files
- `logs/simulation_YYYYMMDD_HHMMSS.log`: Detailed simulation logs
- `logs/recruitment_decisions.log`: Dynamic selection decisions
- `logs/vision_processing.log`: Vision-related processing logs
- `logs/error_recovery.log`: Error handling and recovery attempts

## 🚨 Troubleshooting

### Common Issues

#### Dynamic Selection Not Working
```bash
# Check if explicit configs are disabling it
python dataset_runner.py --dataset medqa --enable-dynamic-selection  # Good
python dataset_runner.py --dataset medqa --leadership --enable-dynamic-selection  # Bad - leadership disables it
```

#### Vision Processing Errors
```bash
# Check image validation
2024-01-01 [WARNING] PMC-VQA question has invalid image, setting image to None
2024-01-01 [ERROR] Vision-related error for Medical Generalist, falling back to text-only
```

#### Memory Issues with Large Datasets
```bash
# Reduce parallel processing
export MAX_WORKERS=2
python dataset_runner.py --dataset ddxplus --num-questions 100
```

### Error Categories
- `vision_error`: Image processing failures
- `recruitment_error`: Agent recruitment failures  
- `timeout_error`: Processing timeout
- `api_error`: Deployment/API failures
- `validation_error`: Ground truth validation failures


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
