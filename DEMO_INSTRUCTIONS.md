# Big5-Agents Demo Instructions

This directory contains demonstration scripts to showcase the Big5-Agents multi-agent system capabilities for interviews and presentations.

## 🎯 Demo Files Overview

### 1. `system_check.bat` 
**⚠️ Run this first!** Validates system setup and configuration.
- Checks Python installation and dependencies
- Validates configuration files
- Tests basic system functionality
- **Duration:** 30 seconds

### 2. `quick_demo.bat`
**⏰ For time-constrained presentations** (interviews, quick demos).
- 3 focused demonstrations with minimal questions
- Perfect for interviews with limited time
- **Duration:** 2-3 minutes

### 3. `demo_showcase.bat`
**🎪 For comprehensive presentations** (detailed demonstrations).
- 8 comprehensive scenarios showcasing all capabilities
- Complete system functionality demonstration
- **Duration:** 15-20 minutes

## 📁 Demo Results Organization

All demo results are organized in the `./demo/` directory:

```
demo/
├── ranking_task/              # Demo 1: NASA Lunar Survival
├── medmcqa_teamwork/         # Demo 2: Medical MCQ with teamwork
├── vision_advanced/          # Demo 3: PMC-VQA with advanced features
├── clinical_diagnosis/       # Demo 4: PubMedQA with shared models
├── trust_collaboration/      # Demo 5: MedBullets with mutual trust
├── comprehensive_comparison/ # Demo 6: All configurations comparison
├── dynamic_selection/        # Demo 7: Dynamic AI-driven config
├── static_config/           # Demo 7: Static configuration comparison
├── pathology_analysis/      # Demo 8: Path-VQA specialized analysis
├── quick_medical/           # Quick demo: Medical questions
├── quick_vision/            # Quick demo: Vision analysis
└── quick_ranking/           # Quick demo: Ranking task
```

Each demo folder contains:
- **Summary results** (JSON files with performance metrics)
- **Detailed agent logs** (individual agent reasoning)
- **Team interaction logs** (collaboration patterns)
- **Decision aggregation results** (voting outcomes)

## 🚀 Usage Instructions

### Pre-Demo Setup
1. **Configure API keys** in `.env` file
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run system check:** `system_check.bat`

### For Interview Presentations
```bash
# Quick validation
system_check.bat

# 2-3 minute demonstration
quick_demo.bat
```

### For Detailed Presentations
```bash
# System validation
system_check.bat

# Complete showcase (15-20 minutes)
demo_showcase.bat
```

## 📋 Demo Scenarios Detailed

### Quick Demo (`quick_demo.bat`)

| Demo | Dataset | Questions | Features | Output Directory |
|------|---------|-----------|----------|------------------|
| **Medical Collaboration** | MedMCQA | 2 | Team Leadership + Medical Recruitment | `demo/quick_medical/` |
| **Vision Analysis** | PMC-VQA | 1 | Advanced Teamwork + Vision Processing | `demo/quick_vision/` |
| **Classic Decision** | NASA Lunar | 1 task | Team Leadership + Intermediate Recruitment | `demo/quick_ranking/` |

### Full Showcase (`demo_showcase.bat`)

| Demo | Dataset | Questions | Key Features | Output Directory |
|------|---------|-----------|--------------|------------------|
| **1. Ranking Task** | NASA Lunar Survival | 1 task | Team Leadership, Agent Recruitment | `demo/ranking_task/` |
| **2. Medical MCQ** | MedMCQA | 5 | Closed-loop Communication, Mutual Monitoring | `demo/medmcqa_teamwork/` |
| **3. Vision Analysis** | PMC-VQA | 3 | All Big5 Components, Advanced Recruitment | `demo/vision_advanced/` |
| **4. Clinical Diagnosis** | PubMedQA | 4 | Shared Mental Models, Team Orientation | `demo/clinical_diagnosis/` |
| **5. Trust Collaboration** | MedBullets | 3 | Mutual Trust (0.9 factor), Balanced Teams | `demo/trust_collaboration/` |
| **6. Configuration Comparison** | MedQA | 10 | All Configurations vs Baseline | `demo/comprehensive_comparison/` |
| **7a. Dynamic Selection** | DDXPlus | 5 | AI-Driven Team Configuration | `demo/dynamic_selection/` |
| **7b. Static Configuration** | DDXPlus | 5 | Fixed Leadership + Communication | `demo/static_config/` |
| **8. Pathology Analysis** | Path-VQA | 2 | Complete Big5, Vision Specialists | `demo/pathology_analysis/` |

## 🎯 Key Capabilities Demonstrated

### 🧠 Big Five Teamwork Model
- **Team Leadership:** Coordination and task definition
- **Closed-Loop Communication:** Feedback and clarification
- **Mutual Monitoring:** Performance tracking and assistance
- **Shared Mental Models:** Common understanding development
- **Team Orientation:** Collective goal alignment
- **Mutual Trust:** Confidence-weighted collaboration

### 🔧 Agent Recruitment Strategies
- **Basic:** Single generalist agent
- **Intermediate:** 3-4 specialized agents
- **Advanced:** 5+ agents with hierarchical organization
- **Medical Pool:** Domain-specific specialists
- **Vision-Enabled:** Image analysis capabilities

### 📊 Decision Aggregation Methods
- **Majority Voting:** Democratic decision making
- **Weighted Voting:** Expertise-based weighting
- **Borda Count:** Ranked preference aggregation

### 🏥 Medical Domain Coverage
- **MedMCQA:** Indian medical entrance exams
- **MedQA:** US medical licensing (USMLE)
- **PubMedQA:** Research-based clinical questions
- **PMC-VQA:** Medical image analysis from papers
- **Path-VQA:** Pathology slide interpretation
- **MedBullets:** Clinical case scenarios
- **DDXPlus:** Diagnostic case studies

## 📈 Interpreting Demo Results

### Result Files in Each Demo Directory

#### `summary.json`
```json
{
  "accuracy": {
    "majority_voting": 0.85,
    "weighted_voting": 0.90,
    "borda_count": 0.87
  },
  "team_composition": ["Cardiologist", "Neurologist", "Radiologist"],
  "teamwork_components": ["leadership", "closed_loop", "mutual_monitoring"]
}
```

#### `detailed_results_enhanced.json`
- Individual question analysis
- Agent reasoning traces
- Team interaction patterns
- Decision evolution process

#### `agent_logs/`
- Detailed agent conversations
- Reasoning explanations
- Confidence assessments
- Collaboration patterns

### Performance Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Accuracy** | Percentage of correct answers | > 0.80 |
| **Confidence** | Agent certainty in decisions | 0.7 - 0.9 |
| **Agreement** | Inter-agent consensus level | > 0.6 |
| **Response Time** | Processing duration | < 60s per question |

## 🔍 Demo Highlights for Presentations

### For Technical Audiences
1. **Show `demo/comprehensive_comparison/`** - Performance across configurations
2. **Highlight `demo/vision_advanced/`** - Multi-modal capabilities
3. **Display `demo/dynamic_selection/`** vs `demo/static_config/` - Adaptive intelligence

### For Medical Audiences
1. **Focus on `demo/clinical_diagnosis/`** - Clinical reasoning
2. **Showcase `demo/pathology_analysis/`** - Specialist collaboration
3. **Demonstrate `demo/medmcqa_teamwork/`** - Medical knowledge integration

### For Business Audiences
1. **Present `demo/trust_collaboration/`** - Team dynamics
2. **Show `demo/comprehensive_comparison/`** - ROI of teamwork
3. **Highlight scalability and efficiency metrics

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### ❌ API Key Errors
```
Error: OpenAI API key not found
Solution: Configure .env file with proper API keys
```

#### ❌ Dataset Download Failures
```
Error: Failed to load dataset
Solution: Check internet connection, run system_check.bat
```

#### ❌ Missing Dependencies
```
Error: Module not found
Solution: pip install -r requirements.txt
```

#### ❌ Permission Errors
```
Error: Cannot write to demo directory
Solution: Run as administrator or check folder permissions
```

### Performance Issues

#### Slow Execution
- **Reduce question count** in bat files
- **Check API rate limits** in deployments
- **Verify internet connectivity**

#### High Memory Usage
- **Close other applications**
- **Reduce team size** (--n-max parameter)
- **Use sequential processing**

## 🎓 Educational Use

### For Students
- Run individual demos to understand different concepts
- Examine agent logs to see reasoning processes
- Compare different teamwork configurations

### For Researchers
- Use comprehensive comparison results
- Analyze decision aggregation effectiveness
- Study multi-modal processing capabilities

### For Developers
- Examine code integration patterns
- Study API usage and rate limiting
- Understand modular component architecture

## 📞 Support and Next Steps

### After Running Demos
1. **Review results** in respective demo directories
2. **Examine logs** for detailed agent reasoning
3. **Compare configurations** using summary files
4. **Customize parameters** for specific use cases

### For Further Development
- Modify `config.py` for new domains
- Add custom agent roles in recruitment pools
- Implement new decision aggregation methods
- Extend to additional medical datasets

### Research Context
**Paper:** "TeamMedAgents: Enhancing Medical Decision-Making Through Structured Teamwork"
**Framework:** Big Five teamwork model (Salas et al., 2005)
**Focus:** Medical AI collaboration and collective intelligence

---

## 🔗 Quick Reference Commands

```bash
# System validation
system_check.bat

# Quick 3-minute demo
quick_demo.bat  

# Full 20-minute showcase  
demo_showcase.bat

# View results
dir demo\
type demo\[scenario]\summary.json
```

**💡 Pro Tip:** Always run `system_check.bat` first to ensure smooth demonstration execution!