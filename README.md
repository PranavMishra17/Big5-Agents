# Big5-Agents

**First Iteration — TeamMedAgents Research Project**

This repository is the first iteration of the TeamMedAgents research project. The core ideas developed here — modular Big Five teamwork components, dynamic agent recruitment, and multi-dataset medical QA evaluation — were carried forward and refined into the second iteration: [SLM-TeamMedAgents](https://github.com/PranavMishra17/SLM-TeamMedAgents), which is the version on which the published paper is based.

**Published Paper (based on SLM-TeamMedAgents, the second iteration):**
TeamMedAgents: Enhancing Medical Decision-Making Through Structured Teamwork — [arxiv.org/abs/2508.08115](https://arxiv.org/abs/2508.08115)

**Second Iteration Repository:** [github.com/PranavMishra17/SLM-TeamMedAgents](https://github.com/PranavMishra17/SLM-TeamMedAgents)

---

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.08115-red.svg)](https://arxiv.org/abs/2508.08115)
[![Status](https://img.shields.io/badge/Status-First%20Iteration-orange.svg)](#)

<img src="https://github.com/PranavMishra17/Big5-Agents/blob/186d3f604b5597ce5f5f6ebfb020279e55b742c3/frame.png" alt="Big5-Agents Framework" width="800"/>

*A flexible multi-agent system with modular teamwork components based on the Big Five teamwork model. Agents collaborate on various tasks with configurable teamwork behaviors.*

---

## Overview

This system implements a multi-agent approach for collaborative problem solving with the following features:

- Dynamic task handling (ranking, MCQ, open-ended questions)
- Modular teamwork components that can be toggled independently
- Specialized agent roles with distinct expertise
- Multiple decision aggregation methods
- Comprehensive logging system with separate channels
- Performance evaluation against ground truth
- Adaptive agent recruitment based on task complexity
- Dataset processing for batch evaluation

The system is built around the Big Five teamwork model components.

## Big Five Teamwork Model

Our system is built on the Big Five teamwork model introduced by Salas et al. (2005), which identifies five core components of effective teamwork:

1. **Team Leadership** — Coordinating activities and defining approaches
2. **Mutual Performance Monitoring** — Tracking teammates' work and providing feedback
3. **Backup Behavior** — Providing support via specialized knowledge
4. **Adaptability** — Adjusting strategies based on information exchange
5. **Team Orientation** — Shared mental models for better coordination

These components are augmented by three coordinating mechanisms:

6. **Shared Mental Models**
7. **Closed-Loop Communication**
8. **Mutual Trust**

## Results

<div align="center">

<img src="https://github.com/PranavMishra17/Big5-Agents/blob/main/metrics.png" alt="Performance Metrics" width="800"/>

*Performance comparison across different teamwork configurations and medical datasets. The Big5-Agents system demonstrates significant improvements in accuracy and decision-making quality when all teamwork components are enabled.*

</div>

## Installation

```bash
git clone https://github.com/PranavMishra17/Big5-Agents.git
cd Big5-Agents
pip install -r requirements.txt
cp .env.example .env
```

## Usage

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

## Dynamic Features

### Dynamic Teamwork Configuration

- Automatic component selection: AI selects up to 3 teamwork components based on question analysis
- Question-adaptive: each question is analyzed to determine optimal collaboration strategy
- Component options: Leadership, Closed-Loop Communication, Mutual Monitoring, Shared Mental Model, Team Orientation, Mutual Trust

### Dynamic Agent Recruitment

- Adaptive team size: automatically determines optimal team size (2–5 agents) per question
- Complexity-based: team size and composition adapt to question complexity and scope
- Efficiency optimized: balances expertise diversity with coordination overhead

```bash
# Run with dynamic selection (default)
python dataset_runner.py --dataset medqa --num-questions 20

# Disable dynamic selection and use static config
python dataset_runner.py --dataset pubmedqa --num-questions 30 --disable-dynamic-selection --leadership --closedloop

# Run all configurations including dynamic
python dataset_runner.py --dataset medbullets --num-questions 40 --all
```

## Supported Datasets

### Text-Only
- MedQA, MedMCQA, PubMedQA, MMLU-Pro Medical, DDXPlus, MedBullets

### Vision-Enabled
- PMC-VQA, Path-VQA

## Configuration Options

```bash
--enable-dynamic-selection    # Enable AI-driven team configuration (default: True)
--disable-dynamic-selection   # Use static configuration only

--leadership              # Enable team leadership
--closedloop              # Enable closed-loop communication
--mutual                  # Enable mutual monitoring
--mental                  # Enable shared mental model
--orientation             # Enable team orientation
--trust                   # Enable mutual trust
--trust-factor 0.8        # Set mutual trust factor (0.0-1.0)

--recruitment                    # Enable dynamic agent recruitment
--recruitment-method adaptive    # Method: adaptive, basic, intermediate, advanced
--recruitment-pool medical       # Pool: general, medical
--n-max 5                        # Maximum agents (disables dynamic sizing if set)

--dataset medqa              # Dataset to use
--num-questions 50           # Number of questions to process
--seed 42                    # Random seed for reproducibility
--all                        # Run all configurations for comparison
--output-dir ./results       # Output directory for results
```

## Project Structure

```
big5_agents/
├── components/
│   ├── agent.py
│   ├── modular_agent.py
│   ├── agent_recruitment.py
│   ├── closed_loop.py
│   ├── mutual_monitoring.py
│   ├── shared_mental_model.py
│   ├── decision_methods.py
│   ├── team_orientation.py
│   └── mutual_trust.py
├── utils/
│   ├── logger.py
│   └── prompts.py
├── simulator.py
├── config.py
├── main.py
├── dataset_runner.py
├── requirements.txt
└── README.md
```

## Logs and Output Structure

```
logs/
├── baseline/
├── leadership/
├── recruitment_basic/
├── recruitment_intermediate/
└── leadership_closed_loop_mutual_monitoring_shared_mental_model_recruitment/

output/
├── sim_YYYYMMDD_HHMMSS_results.json
├── all_configurations_results.json
└── medqa_results/
    ├── baseline/
    ├── recruitment_adaptive/
    └── combined_results.json
```

## References

- Kim, Y. et al. (2024). MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making. [arXiv:2404.15155](https://arxiv.org/abs/2404.15155)
- Salas, E., Sims, D. E., & Burke, C. S. (2005). Is there a "Big Five" in Teamwork? Small Group Research, 36(5), 555–599.
