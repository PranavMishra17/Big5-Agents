"""
agent_recruitment.py with dynamic teamwork configuration and agent count selection.
Maintains backward compatibility while adding LLM-driven dynamic selection.
"""
import logging
import random
import traceback
from typing import Dict, List, Any, Tuple, Optional

from components.agent import Agent
from components.modular_agent import MedicalImageAnalyst, ModularAgent, PathologySpecialist
import config

from utils.prompts import RECRUITMENT_PROMPTS, DYNAMIC_RECRUITMENT_PROMPTS

import threading
thread_local = threading.local()

# Global counters for complexity tracking
complexity_counts = {
    "basic": 0,
    "intermediate": 0, 
    "advanced": 0
}

# Track correct answers by complexity
complexity_correct = {
    "basic": 0,
    "intermediate": 0,
    "advanced": 0
}

def reset_complexity_metrics():
    """Reset complexity metrics counters."""
    global complexity_counts, complexity_correct
    complexity_counts = {"basic": 0, "intermediate": 0, "advanced": 0}
    complexity_correct = {"basic": 0, "intermediate": 0, "advanced": 0}

def determine_complexity(question, method="adaptive"):
    """
    Determine the complexity of a question to decide on team structure.
    
    Args:
        question: The question or task to analyze
        method: Method for complexity determination
        
    Returns:
        Complexity level ("basic", "intermediate", or "advanced")
    """
    global complexity_counts
    
    if method in ["basic", "intermediate", "advanced"]:
        complexity = method
    else:
        # For adaptive method, try evaluation with error handling
        try:
            evaluator = Agent(
                role="Complexity Evaluator",
                expertise_description="analyzes tasks to determine their complexity level"
            )
    
            prompt = RECRUITMENT_PROMPTS["complexity_evaluation"].format(
                question=question
            )

            try:
                response = evaluator.chat(prompt)
                
                # Extract complexity classification
                if "1)" in response.lower() or "low" in response.lower() or "basic" in response.lower():
                    complexity = "basic"
                elif "2)" in response.lower() or "moderate" in response.lower() or "intermediate" in response.lower():
                    complexity = "intermediate"
                else:
                    complexity = "advanced"
                    
                logging.info(f"Complexity determination: {complexity}")
            except Exception as e:
                # API error fallback: use heuristic approach
                logging.error(f"Error in complexity evaluation: {str(e)}")
                
                # Simple heuristic based on length and keywords
                word_count = len(question.split())
                complex_terms = ["autoimmune", "encephalitis", "differential", 
                                "pathophysiology", "etiology", "comorbidities"]
                term_count = sum(1 for term in complex_terms if term.lower() in question.lower())
                
                if word_count > 100 or term_count >= 3:
                    complexity = "advanced"
                elif word_count > 50 or term_count >= 1:
                    complexity = "intermediate"
                else:
                    complexity = "basic"
                logging.info(f"Used fallback complexity determination: {complexity}")
        except Exception as e:
            # Default if all else fails
            logging.error(f"Critical error in complexity determination, using default: {str(e)}")
            logging.error(traceback.format_exc())
            complexity = "intermediate"
    
    # Update counter
    complexity_counts[complexity] += 1
    return complexity

def determine_optimal_team_size(question: str, complexity: str, max_agents: int = 5) -> int:
    """
    Dynamically determine optimal team size based on question complexity and content.
    
    Args:
        question: The question to analyze
        complexity: Pre-determined complexity level
        max_agents: Maximum number of agents allowed
        
    Returns:
        Optimal number of agents (2-5)
    """
    try:
        evaluator = Agent(
            role="Team Size Optimizer",
            expertise_description="determines optimal team size for collaborative problem-solving"
        )
        
        prompt = DYNAMIC_RECRUITMENT_PROMPTS["team_size_determination"].format(
            question=question,
            complexity=complexity,
            max_agents=max_agents
        )
        
        response = evaluator.chat(prompt)
        
        # Extract team size from response
        import re
        size_match = re.search(r"TEAM_SIZE:\s*(\d+)", response)
        
        if size_match:
            optimal_size = int(size_match.group(1))
            # Ensure within bounds
            optimal_size = max(2, min(optimal_size, max_agents))
            logging.info(f"Dynamic team size determination: {optimal_size} agents")
            return optimal_size
        else:
            # Fallback based on complexity
            fallback_sizes = {"basic": 2, "intermediate": 3, "advanced": 4}
            fallback_size = fallback_sizes.get(complexity, 3)
            logging.warning(f"Could not extract team size, using fallback: {fallback_size}")
            return fallback_size
            
    except Exception as e:
        logging.error(f"Error in team size determination: {str(e)}")
        # Complexity-based fallback
        fallback_sizes = {"basic": 2, "intermediate": 3, "advanced": 4}
        return fallback_sizes.get(complexity, 3)

def determine_optimal_teamwork_config(question: str, complexity: str, team_size: int) -> Dict[str, bool]:
    """
    Dynamically determine optimal teamwork configuration based on question and team characteristics.
    Maximum 3 components will be selected.
    
    Args:
        question: The question to analyze
        complexity: Pre-determined complexity level
        team_size: Number of agents in the team
        
    Returns:
        Dictionary with teamwork component selections
    """
    try:
        evaluator = Agent(
            role="Teamwork Configuration Specialist",
            expertise_description="determines optimal teamwork components for collaborative problem-solving"
        )
        
        prompt = DYNAMIC_RECRUITMENT_PROMPTS["teamwork_config_selection"].format(
            question=question,
            complexity=complexity,
            team_size=team_size
        )
        
        response = evaluator.chat(prompt)
        
        # Extract selected components
        selected_components = []
        component_mapping = {
            "leadership": "use_team_leadership",
            "closed_loop": "use_closed_loop_comm", 
            "monitoring": "use_mutual_monitoring",
            "mental_model": "use_shared_mental_model",
            "orientation": "use_team_orientation",
            "trust": "use_mutual_trust"
        }
        
        # Parse response for selected components
        import re
        selected_match = re.search(r"SELECTED_COMPONENTS:\s*(.+)", response)
        
        if selected_match:
            selected_text = selected_match.group(1).lower()
            for component, config_key in component_mapping.items():
                if component in selected_text:
                    selected_components.append(config_key)
        
        # Limit to maximum 3 components
        selected_components = selected_components[:3]
        
        # Create configuration dictionary
        teamwork_config = {
            "use_team_leadership": False,
            "use_closed_loop_comm": False,
            "use_mutual_monitoring": False,
            "use_shared_mental_model": False,
            "use_team_orientation": False,
            "use_mutual_trust": False
        }
        
        # Enable selected components
        for component in selected_components:
            teamwork_config[component] = True
        
        selected_names = [k.replace("use_", "").replace("_", " ") for k, v in teamwork_config.items() if v]
        logging.info(f"Dynamic teamwork configuration: {selected_names}")
        
        return teamwork_config
        
    except Exception as e:
        logging.error(f"Error in teamwork configuration determination: {str(e)}")
        # Fallback based on complexity and team size
        if complexity == "basic":
            return {
                "use_team_leadership": True,
                "use_closed_loop_comm": False,
                "use_mutual_monitoring": False,
                "use_shared_mental_model": False,
                "use_team_orientation": False,
                "use_mutual_trust": False
            }
        elif complexity == "intermediate":
            return {
                "use_team_leadership": True,
                "use_closed_loop_comm": True,
                "use_mutual_monitoring": team_size > 3,
                "use_shared_mental_model": False,
                "use_team_orientation": False,
                "use_mutual_trust": False
            }
        else:  # advanced
            return {
                "use_team_leadership": True,
                "use_closed_loop_comm": True,
                "use_mutual_monitoring": True,
                "use_shared_mental_model": False,
                "use_team_orientation": False,
                "use_mutual_trust": False
            }

def recruit_agents_isolated(question: str, complexity: str, recruitment_pool: str = "general", 
                          n_max: Optional[int] = None, recruitment_method: str = "adaptive", 
                          deployment_config=None, task_config=None, teamwork_config=None,
                          enable_dynamic_selection: bool = True):
    """
    Enhanced recruitment with dynamic team size and teamwork configuration.
    Maintains backward compatibility.

    Args:
        question: The question or task to analyze
        complexity: Determined complexity level
        recruitment_pool: Pool of agent types to recruit from
        n_max: Maximum number of agents (if None, uses dynamic selection)
        recruitment_method: Method for agent recruitment
        deployment_config: Specific deployment configuration
        task_config: Isolated task configuration
        teamwork_config: Teamwork configuration (if None, uses dynamic selection)
        enable_dynamic_selection: Whether to use dynamic selection (backward compatibility)

    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    import threading
    from components.modular_agent import ModularAgent

    # Add instance ID to avoid agent reuse per thread/question
    try:
        instance_id = id(question) + thread_local.question_index
        logging.debug(f"Using thread-local question index for instance ID: {thread_local.question_index}")
    except Exception:
        instance_id = id(question)
        logging.debug("Thread-local question index not found; falling back to question ID.")

    # BACKWARD COMPATIBILITY: If specific configs provided, skip dynamic selection
    use_dynamic_team_size = (n_max is None and enable_dynamic_selection)
    use_dynamic_teamwork = (teamwork_config is None and enable_dynamic_selection)
    
    # Determine team size
    if use_dynamic_team_size:
        optimal_team_size = determine_optimal_team_size(question, complexity, max_agents=5)
        logging.info(f"Dynamic team size selection: {optimal_team_size} agents")
    else:
        optimal_team_size = n_max if n_max is not None else 5
        logging.info(f"Using provided team size: {optimal_team_size} agents")
    
    # Determine teamwork configuration
    if use_dynamic_teamwork:
        optimal_teamwork_config = determine_optimal_teamwork_config(question, complexity, optimal_team_size)
        logging.info(f"Dynamic teamwork configuration selected")
    else:
        optimal_teamwork_config = teamwork_config
        logging.info(f"Using provided teamwork configuration")

    logging.info(f"Recruiting agents using method: {recruitment_method}, complexity: {complexity}, "
                f"team_size: {optimal_team_size}, dynamic_selection: {enable_dynamic_selection}")

    # Enhanced vision detection
    has_image = False
    if task_config and "image_data" in task_config:
        image_data = task_config["image_data"]
        has_image = (
            image_data.get("image_available", False) and 
            image_data.get("image") is not None and
            image_data.get("requires_visual_analysis", False)
        )
        # Additional check for PMC-VQA dataset
        metadata = task_config.get("metadata", {})
        if metadata.get("dataset") == "pmc_vqa":
            has_image = True
            logging.info("PMC-VQA dataset detected - enabling vision recruitment")
    
    # If this is a vision task, use specialized recruitment
    if has_image:
        logging.info(f"Vision task detected - using specialized vision-capable recruitment")
        return recruit_vision_capable_agents(
            question=question,
            has_image=has_image,
            complexity=complexity,
            deployment_config=deployment_config,
            task_config=task_config,
            teamwork_config=optimal_teamwork_config
        )

    # Route to appropriate recruitment method
    if complexity == "basic" or recruitment_method == "basic":
        logging.info("Using basic recruitment (single agent)")
        return recruit_basic_team_isolated(question, recruitment_pool, deployment_config, 
                                         task_config, instance_id=instance_id)
    elif complexity == "intermediate" or recruitment_method == "intermediate":
        logging.info(f"Using intermediate recruitment with team_size={optimal_team_size}")
        return recruit_intermediate_team_isolated(question, recruitment_pool, optimal_team_size, 
                                                deployment_config, task_config, instance_id=instance_id,
                                                teamwork_config=optimal_teamwork_config)
    elif complexity == "advanced" or recruitment_method == "advanced":
        logging.info("Using advanced recruitment (MDT structure)")
        return recruit_advanced_team_isolated(question, recruitment_pool, deployment_config, 
                                            task_config, optimal_teamwork_config, instance_id=instance_id)
    else:
        logging.info(f"Unrecognized complexity/method: {complexity}/{recruitment_method}, falling back to intermediate")
        return recruit_intermediate_team_isolated(question, recruitment_pool, optimal_team_size, 
                                                deployment_config, task_config, instance_id=instance_id,
                                                teamwork_config=optimal_teamwork_config)

def recruit_intermediate_team_isolated(question: str, recruitment_pool: str, n_max: int = 5, 
                                     deployment_config=None, task_config=None, instance_id=None,
                                     teamwork_config=None):
    """
    Enhanced intermediate team recruitment with dynamic teamwork configuration.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        n_max: Number of agents to recruit
        deployment_config: Specific deployment configuration
        task_config: Isolated task configuration
        instance_id: Instance identifier
        teamwork_config: Teamwork configuration to apply
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    from components.modular_agent import ModularAgent
    
    # Use provided teamwork config or fallback to global config
    if teamwork_config is None:
        teamwork_config = {
            "use_team_leadership": config.USE_TEAM_LEADERSHIP,
            "use_closed_loop_comm": config.USE_CLOSED_LOOP_COMM,
            "use_mutual_monitoring": config.USE_MUTUAL_MONITORING,
            "use_shared_mental_model": config.USE_SHARED_MENTAL_MODEL,
            "use_team_orientation": config.USE_TEAM_ORIENTATION,
            "use_mutual_trust": config.USE_MUTUAL_TRUST
        }
    
    # Log recruitment parameters
    logging.info(f"Intermediate team recruitment: n_max={n_max}, pool={recruitment_pool}")
    
    # Create a recruiter agent
    recruiter = Agent(
        role="Recruiter",
        expertise_description="Assembles teams of medical experts for collaborative problem-solving"
    )
    
    # Create prompt for team selection
    selection_prompt = RECRUITMENT_PROMPTS["team_selection"].format(
        question=question,
        num_agents=n_max
    )

    response = recruiter.chat(selection_prompt)
    
    # Parse selected team
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    selected_team = []
    leader_role = None
    hierarchies = {}
    weights = {}
    
    for line in lines:
        if " - Hierarchy: " not in line:
            continue
            
        # Extract role, expertise, and hierarchy
        parts = line.split(" - Hierarchy: ")
        if len(parts) != 2:
            continue
            
        agent_info, hierarchy_weight = parts
        
        # Extract role and expertise from agent info
        try:
            if '.' in agent_info:
                agent_info = agent_info.split('.', 1)[1].strip()
                
            if ' - ' in agent_info:
                role, expertise = agent_info.split(' - ', 1)
            else:
                role = agent_info
                expertise = "General expertise in this domain"
        except:
            continue
            
        # Extract weight
        weight = 0.2  # Default weight
        if " - Weight: " in hierarchy_weight:
            hierarchy, weight_part = hierarchy_weight.split(" - Weight: ")
            try:
                weight = float(weight_part.strip())
            except:
                weight = 0.2
        else:
            hierarchy = hierarchy_weight
        
        # Determine hierarchy
        if "Independent" in hierarchy:
            hierarchies[role] = "Independent"
        else:
            if ">" in hierarchy:
                parts = hierarchy.split(">")
                superior = parts[0].strip()
                if superior == role:
                    if leader_role is None:
                        leader_role = role
        
        # Store weight and add to selected team
        weights[role] = weight
        selected_team.append((role, expertise, hierarchies.get(role, "Independent"), weight))
        
        # Stop when we have enough agents
        if len(selected_team) >= n_max:
            break
    
    # Apply n_max constraint
    if len(selected_team) > n_max:
        logging.info(f"Limiting team size from {len(selected_team)} to {n_max}")
        selected_team = selected_team[:n_max]
    
    # If no team members found or less than specified, create a default team
    if len(selected_team) < n_max:
        default_roles = [
            ("Medical Generalist", "Broad knowledge across medical disciplines", "Leader", 0.3),
            ("Medical Specialist", "Focused expertise in relevant area", "Independent", 0.3),
            ("Diagnostician", "Expert in diagnostic reasoning", "Independent", 0.4)
        ]
        
        for role_info in default_roles:
            if len(selected_team) >= n_max:
                break
            if not any(role_info[0] == team_role[0] for team_role in selected_team):
                selected_team.append(role_info)
        
        if not leader_role:
            leader_role = "Medical Generalist"
    
    # If no leader designated, choose the first as leader
    if not leader_role:
        leader_role = selected_team[0][0]
        selected_team[0] = (leader_role, selected_team[0][1], "Leader", selected_team[0][3])
    
    # Create agents with teamwork configuration
    agents = {}
    leader = None
    agent_index = 0
    
    for role, expertise, hierarchy, weight in selected_team:
        is_leader = hierarchy == "Leader" or role == leader_role
        
        # Get deployment config for this agent or use provided one
        if deployment_config:
            agent_deployment_config = deployment_config
        else:
            agent_deployment_config = config.get_deployment_for_agent(agent_index)
        
        agent = ModularAgent(
            role_type=role,
            use_team_leadership=is_leader and teamwork_config.get("use_team_leadership", False),
            use_closed_loop_comm=teamwork_config.get("use_closed_loop_comm", False),
            use_mutual_monitoring=teamwork_config.get("use_mutual_monitoring", False),
            use_shared_mental_model=teamwork_config.get("use_shared_mental_model", False),
            use_team_orientation=teamwork_config.get("use_team_orientation", False),
            use_mutual_trust=teamwork_config.get("use_mutual_trust", False),
            deployment_config=agent_deployment_config,
            agent_index=agent_index,
            task_config=task_config
        )
        
        # Store weight in agent's knowledge base
        agent.add_to_knowledge_base("weight", weight)
        
        agents[role] = agent
        
        if is_leader:
            leader = agent
            
        agent_index += 1
    
    # Log recruitment with teamwork configuration
    deployment_info = [f"{role}:{agent.deployment_config['name']}" for role, agent in agents.items()]
    teamwork_enabled = [name for name, enabled in teamwork_config.items() if enabled]
    
    logging.info(f"Intermediate complexity: Recruited team of {len(agents)} experts with leader '{leader_role}'")
    logging.info(f"Agent weights: {weights}")
    logging.info(f"Teamwork components enabled: {teamwork_enabled}")
    logging.info(f"Deployment distribution: {deployment_info}")
    
    return agents, leader

# Keep existing functions for backward compatibility
def recruit_agents(question: str, complexity: str, recruitment_pool: str = "general", 
                  n_max=5, recruitment_method: str = "adaptive"):
    """Legacy recruit_agents function that uses global config."""
    return recruit_agents_isolated(question, complexity, recruitment_pool, n_max, 
                                 recruitment_method, None, config.TASK)

def recruit_basic_team_isolated(question: str, recruitment_pool: str, deployment_config=None, 
                              task_config=None, instance_id=None):
    """
    Recruit a single medical generalist for basic questions using isolated configuration.
    Always returns exactly ONE agent, regardless of n_max.
    """
    from components.modular_agent import ModularAgent
    
    role = "Medical Generalist"
    expertise = "A general medical practitioner with broad knowledge across medical disciplines"
    
    # Get deployment config for agent 0 or use provided one
    if deployment_config:
        agent_deployment_config = deployment_config
    else:
        agent_deployment_config = config.get_deployment_for_agent(0)
    
    # Create the generalist agent
    agent = ModularAgent(
        role_type=role, 
        use_team_leadership=False,  
        use_closed_loop_comm=False,
        use_mutual_monitoring=False,
        use_shared_mental_model=False,
        use_team_orientation=False,
        use_mutual_trust=False,
        deployment_config=agent_deployment_config,
        agent_index=0,
        task_config=task_config
    )
    
    # Create the agents dictionary
    agents = {"Medical Generalist": agent}
    
    logging.info(f"Basic recruitment: Created a single Medical Generalist with deployment {agent_deployment_config['name']}")
    
    return agents, agent

def recruit_advanced_team_isolated(question: str, recruitment_pool: str, deployment_config=None, 
                                 task_config=None, teamwork_config=None, instance_id=None):
    """
    Recruit multiple specialized teams for advanced questions with teamwork configuration.
    """
    from components.modular_agent import ModularAgent
    
    # Use provided teamwork config or fallback to global config
    if teamwork_config is None:
        teamwork_config = {
            "use_team_leadership": config.USE_TEAM_LEADERSHIP,
            "use_closed_loop_comm": config.USE_CLOSED_LOOP_COMM,
            "use_mutual_monitoring": config.USE_MUTUAL_MONITORING,
            "use_shared_mental_model": config.USE_SHARED_MENTAL_MODEL,
            "use_team_orientation": config.USE_TEAM_ORIENTATION,
            "use_mutual_trust": config.USE_MUTUAL_TRUST
        }
    
    # Create a strategic team designer
    recruiter = Agent(
        role="Strategic Team Designer",
        expertise_description="Designs multi-team systems for complex medical problem solving"
    )
    
    # Create prompt for multi-team design following MDAgents MDT approach
    design_prompt = RECRUITMENT_PROMPTS["mdt_design"].format(
        question=question
    )

    response = recruiter.chat(design_prompt)
    
    # Parse multi-team structure (same as before)
    teams = []
    current_team = None
    chief_coordinator = None
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for new team definition
        if line.startswith('Group') and '-' in line:
            if current_team:
                teams.append(current_team)
                
            team_name = line.split('-', 1)[1].strip()
            current_team = {
                'name': team_name,
                'members': []
            }
        # Check for team member
        elif current_team and line.startswith('Member') and ':' in line:
            parts = line.split(':', 1)[1].strip()
            
            # Check if this member is a leader
            is_leader = '(Lead)' in parts
            
            # Find the role, which is before the first -
            if '-' in parts:
                role_part = parts.split('-', 1)[0].strip()
                expertise = parts.split('-', 1)[1].strip()
            else:
                role_part = parts
                expertise = "Medical expertise"
            
            # Remove the (Lead) designation if present
            role = role_part.replace('(Lead)', '').strip()
            
            # Add to the team
            current_team['members'].append({
                'role': role,
                'expertise': expertise,
                'is_leader': is_leader
            })
            
            # If this is in the FRDT and is a lead, they're the chief coordinator
            if current_team['name'].startswith('Final Review') and is_leader:
                chief_coordinator = role
    
    # Add the last team
    if current_team:
        teams.append(current_team)
    
    # If no teams parsed, create default team structure
    if not teams:
        teams = [
            {
                'name': 'Initial Assessment Team (IAT)',
                'members': [
                    {'role': 'Internist', 'expertise': 'General medical assessment', 'is_leader': True},
                    {'role': 'Radiologist', 'expertise': 'Imaging interpretation', 'is_leader': False},
                    {'role': 'Pathologist', 'expertise': 'Laboratory test interpretation', 'is_leader': False}
                ]
            },
            {
                'name': 'Domain Specialist Team (DST)',
                'members': [
                    {'role': 'Cardiologist', 'expertise': 'Heart and circulatory system', 'is_leader': True},
                    {'role': 'Neurologist', 'expertise': 'Nervous system', 'is_leader': False},
                    {'role': 'Endocrinologist', 'expertise': 'Hormone disorders', 'is_leader': False}
                ]
            },
            {
                'name': 'Final Review and Decision Team (FRDT)',
                'members': [
                    {'role': 'Senior Consultant', 'expertise': 'Medical decision making', 'is_leader': True},
                    {'role': 'Evidence-Based Medicine Specialist', 'expertise': 'Study evaluation', 'is_leader': False},
                    {'role': 'Clinical Decision Maker', 'expertise': 'Treatment selection', 'is_leader': False}
                ]
            }
        ]
        chief_coordinator = 'Senior Consultant'
    
    # Create all agents with teamwork configuration
    agents = {}
    leader = None
    agent_index = 0
    
    # Use a prefix to make roles unique across teams
    for team_idx, team in enumerate(teams):
        team_prefix = f"{team_idx+1}_{team['name'].split('(')[0].strip()}_"
        
        for member in team['members']:
            role = member['role']
            expertise = member['expertise']
            is_leader = member['is_leader']
            
            # Create unique role identifier
            unique_role = f"{team_prefix}{role}"
            
            # Get deployment config for this agent or use provided one
            if deployment_config:
                agent_deployment_config = deployment_config
            else:
                agent_deployment_config = config.get_deployment_for_agent(agent_index)
            
            agent = ModularAgent(
                role_type=unique_role,
                use_team_leadership=is_leader and teamwork_config.get("use_team_leadership", False),
                use_closed_loop_comm=teamwork_config.get("use_closed_loop_comm", False),
                use_mutual_monitoring=teamwork_config.get("use_mutual_monitoring", False),
                use_shared_mental_model=teamwork_config.get("use_shared_mental_model", False),
                use_team_orientation=teamwork_config.get("use_team_orientation", False),
                use_mutual_trust=teamwork_config.get("use_mutual_trust", False),
                deployment_config=agent_deployment_config,
                agent_index=agent_index,
                task_config=task_config
            )
            
            # Add hierarchical information to knowledge base
            agent.add_to_knowledge_base("hierarchy", {
                "is_leader": is_leader,
                "team": team['name'],
                "is_decision_team": "final" in team['name'].lower() or "frdt" in team['name'].lower()
            })
            
            # Add to agents dictionary
            agents[unique_role] = agent
            
            # Track chief coordinator as overall leader
            if role == chief_coordinator:
                leader = agent
                
            agent_index += 1
    
    # If no chief coordinator found, use first team leader
    if not leader:
        for role, agent in agents.items():
            if hasattr(agent, 'can_lead') and agent.can_lead:
                leader = agent
                break
    
    # Log recruitment with teamwork info
    deployment_info = [f"{role}:{agent.deployment_config['name']}" for role, agent in agents.items()]
    teamwork_enabled = [name for name, enabled in teamwork_config.items() if enabled]
    
    logging.info(f"Advanced complexity: Recruited {len(agents)} experts in {len(teams)} teams")
    logging.info(f"Chief Coordinator: {chief_coordinator if chief_coordinator else 'Not specified'}")
    logging.info(f"Teamwork components enabled: {teamwork_enabled}")
    logging.info(f"Deployment distribution: {deployment_info}")
    
    return agents, leader

def recruit_vision_capable_agents(question: str, has_image: bool, complexity: str, 
                                deployment_config=None, task_config=None, teamwork_config=None):
    """Recruit vision-capable agents with proper task config passing."""
    
    agents = {}
    leader = None
    agent_index = 0
    
    # Use provided teamwork config or fallback
    if teamwork_config is None:
        teamwork_config = {
            "use_team_leadership": config.USE_TEAM_LEADERSHIP,
            "use_closed_loop_comm": config.USE_CLOSED_LOOP_COMM,
            "use_mutual_monitoring": config.USE_MUTUAL_MONITORING,
            "use_shared_mental_model": config.USE_SHARED_MENTAL_MODEL,
            "use_team_orientation": config.USE_TEAM_ORIENTATION,
            "use_mutual_trust": config.USE_MUTUAL_TRUST
        }
    
    # Determine image type
    question_lower = question.lower()
    image_data = task_config.get("image_data", {}) if task_config else {}
    
    is_pathology = (
        image_data.get("is_pathology_image", False) or
        image_data.get("image_type") == "pathology_slide" or
        any(term in question_lower for term in [
            'pathology', 'histology', 'microscopic', 'tissue', 'cell', 'biopsy'
        ])
    )
    
    # Create primary vision specialist
    if is_pathology:
        specialist = PathologySpecialist(
            deployment_config=deployment_config,
            agent_index=agent_index,
            task_config=task_config,
            **teamwork_config
        )
        role_name = "Pathology Specialist"
    else:
        specialist = MedicalImageAnalyst(
            deployment_config=deployment_config,
            agent_index=agent_index,
            task_config=task_config,
            **teamwork_config
        )
        role_name = "Medical Image Analyst"
    
    agents[role_name] = specialist
    leader = specialist
    agent_index += 1
    
    # Add supporting agents for complex cases
    if complexity in ['intermediate', 'advanced']:
        generalist = ModularAgent(
            role_type='Medical Generalist',
            deployment_config=deployment_config,
            agent_index=agent_index,
            task_config=task_config,
            use_team_leadership=False,
            **{k: v for k, v in teamwork_config.items() if k != 'use_team_leadership'}
        )
        agents['Medical Generalist'] = generalist
    
    logging.info(f"Vision team recruited: {list(agents.keys())} for {'pathology' if is_pathology else 'medical imaging'}")
    return agents, leader

# Legacy functions for backward compatibility
def recruit_basic_team(question: str, recruitment_pool: str, instance_id=None):
    """Legacy wrapper that uses global config."""
    return recruit_basic_team_isolated(question, recruitment_pool, None, config.TASK, instance_id)

def recruit_intermediate_team(question: str, recruitment_pool: str, n_max: int = 5, instance_id=None):
    """Legacy wrapper that uses global config."""
    return recruit_intermediate_team_isolated(question, recruitment_pool, n_max, None, config.TASK, instance_id)

def recruit_advanced_team(question: str, recruitment_pool: str, instance_id=None):
    """Legacy wrapper that uses global config."""
    return recruit_advanced_team_isolated(question, recruitment_pool, None, config.TASK, instance_id)