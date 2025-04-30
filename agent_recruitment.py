"""
Agent recruitment module for dynamically assembling specialized agent teams.
"""

import logging
import random
from typing import Dict, List, Any, Tuple

from agent import Agent
from modular_agent import ModularAgent, create_agent_team
import config

def determine_complexity(question: str, fixed_complexity: str = None) -> str:
    """
    Determine the complexity of a given question.
    
    Args:
        question: The question to evaluate
        fixed_complexity: Optional fixed complexity level
        
    Returns:
        Complexity level (basic, intermediate, or advanced)
    """
    if fixed_complexity and fixed_complexity != "adaptive":
        return fixed_complexity
    
    # Create a basic agent to evaluate complexity
    complexity_agent = Agent(
        role="Complexity Evaluator",
        expertise_description="Evaluates the complexity of questions to determine appropriate agent resources"
    )
    
    # Create prompt for complexity assessment
    complexity_prompt = f"""
    Analyze the following task/question and determine its complexity level:
    
    {question}
    
    Please classify the complexity as one of the following:
    1) basic: A single expert can adequately answer this question.
    2) intermediate: A team of specialists with different expertise should collaborate to address this question.
    3) advanced: Multiple teams with different specializations need to coordinate to properly address this question.
    
    Provide your classification and a brief explanation.
    """
    
    response = complexity_agent.chat(complexity_prompt)
    
    # Determine complexity from response
    if "basic" in response.lower() or "1)" in response.lower():
        return "basic"
    elif "intermediate" in response.lower() or "2)" in response.lower():
        return "intermediate"
    elif "advanced" in response.lower() or "3)" in response.lower():
        return "advanced"
    else:
        # Default to intermediate if unclear
        return "intermediate"

def recruit_agents(question: str, complexity: str, recruitment_pool: str = "general") -> Tuple[Dict[str, ModularAgent], ModularAgent]:
    """
    Recruit appropriate agents based on question complexity.
    
    Args:
        question: The question or task to address
        complexity: The complexity level (basic, intermediate, advanced)
        recruitment_pool: The pool of agent types to recruit from
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    if complexity == "basic":
        return recruit_basic_team(question, recruitment_pool)
    elif complexity == "intermediate":
        return recruit_intermediate_team(question, recruitment_pool)
    elif complexity == "advanced":
        return recruit_advanced_team(question, recruitment_pool)
    else:
        # Default to intermediate if unknown complexity
        return recruit_intermediate_team(question, recruitment_pool)

def recruit_basic_team(question: str, recruitment_pool: str) -> Tuple[Dict[str, ModularAgent], ModularAgent]:
    """
    Recruit a single expert for basic questions.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    # Create a recruiter agent to select the best single expert
    recruiter = Agent(
        role="Recruiter",
        expertise_description="Identifies the most appropriate expert for specific tasks"
    )
    
    # Get available agent types
    agent_options = config.RECRUITMENT_POOLS.get(recruitment_pool, config.RECRUITMENT_POOLS["general"])
    agent_options_text = "\n".join([f"{i+1}. {agent}" for i, agent in enumerate(agent_options)])
    
    # Create prompt for expert selection
    selection_prompt = f"""
    Given the following task/question:
    
    {question}
    
    Select the single most appropriate expert from the following list:
    {agent_options_text}
    
    Return just the name and expertise of your selected expert, formatted exactly as it appears in the list.
    For example: "Critical Analyst - Approaches problems with analytical rigor, questioning assumptions and evaluating evidence"
    """
    
    response = recruiter.chat(selection_prompt)
    
    # Extract selected agent
    selected_agent = None
    for agent_option in agent_options:
        if agent_option.lower() in response.lower():
            selected_agent = agent_option
            break
    
    # If no clear selection, use first option
    if not selected_agent:
        selected_agent = agent_options[0]
    
    # Parse role and expertise
    role, expertise = selected_agent.split(" - ", 1)
    
    # Create agent
    agent = ModularAgent(
        role_type=role,
        use_team_leadership=True,  # Single agent is always leader
        use_closed_loop_comm=config.USE_CLOSED_LOOP_COMM,
        use_mutual_monitoring=config.USE_MUTUAL_MONITORING,
        use_shared_mental_model=config.USE_SHARED_MENTAL_MODEL
    )
    
    # Create agent dictionary
    agents = {role: agent}
    
    # Log recruitment
    logging.info(f"Basic complexity: Recruited single expert '{role}'")
    
    return agents, agent

def recruit_intermediate_team(question: str, recruitment_pool: str) -> Tuple[Dict[str, ModularAgent], ModularAgent]:
    """
    Recruit a team of specialists with hierarchical relationships.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    # Create a recruiter agent to select experts
    recruiter = Agent(
        role="Recruiter",
        expertise_description="Assembles teams of experts for collaborative problem-solving"
    )
    
    # Get available agent types
    agent_options = config.RECRUITMENT_POOLS.get(recruitment_pool, config.RECRUITMENT_POOLS["general"])
    agent_options_text = "\n".join([f"{i+1}. {agent}" for i, agent in enumerate(agent_options)])
    
    # Determine number of agents to recruit (default 5)
    num_agents = min(5, len(agent_options))
    
    # Create prompt for team selection
    selection_prompt = f"""
    Given the following task/question:
    
    {question}
    
    Select {num_agents} experts to form a team, assigning a leader and specifying relationships:
    {agent_options_text}
    
    Return your choices in this format:
    1. [Expert Role] - [Expertise] - Hierarchy: [Leader/SubordinateTo:[Other Expert]/Independent]
    2. [Expert Role] - [Expertise] - Hierarchy: [Leader/SubordinateTo:[Other Expert]/Independent]
    ...
    
    Make only one expert the Leader, and ensure that hierarchical relationships are clear.
    """
    
    response = recruiter.chat(selection_prompt)
    
    # Parse selected team
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    selected_team = []
    leader_role = None
    hierarchies = {}
    
    for line in lines:
        if '-' not in line or 'Hierarchy:' not in line:
            continue
            
        # Extract role, expertise, and hierarchy
        parts = line.split(' - Hierarchy: ')
        if len(parts) != 2:
            continue
            
        agent_info, hierarchy = parts
        
        # Extract role and expertise from agent info
        try:
            # Handle numbering and role separation
            if '.' in agent_info:
                agent_info = agent_info.split('.', 1)[1].strip()
                
            if ' - ' in agent_info:
                role, expertise = agent_info.split(' - ', 1)
            else:
                role = agent_info
                expertise = "General expertise in this domain"
        except:
            continue
            
        # Determine hierarchy
        is_leader = 'leader' in hierarchy.lower()
        if is_leader:
            leader_role = role
            hierarchies[role] = 'Leader'
        else:
            # Check if subordinate to another role
            if 'subordinateto:' in hierarchy.lower():
                superior = hierarchy.lower().split('subordinateto:')[1].strip()
                hierarchies[role] = f'SubordinateTo:{superior}'
            else:
                hierarchies[role] = 'Independent'
        
        selected_team.append((role, expertise, hierarchies[role]))
        
        # Stop when we have enough agents
        if len(selected_team) >= num_agents:
            break
    
    # If no team members found, use default team
    if not selected_team:
        selected_team = [
            ("Critical Analyst", "Approaches problems with analytical rigor", "Leader"),
            ("Domain Expert", "Provides specialized knowledge", "Independent"),
            ("Process Facilitator", "Focuses on optimizing the collaborative process", "Independent"),
            ("Creative Strategist", "Offers innovative perspectives", "Independent"),
            ("Systems Thinker", "Analyzes how different components interact", "Independent")
        ]
        leader_role = "Critical Analyst"
    
    # If no leader designated, choose the first as leader
    if not leader_role:
        leader_role = selected_team[0][0]
        selected_team[0] = (leader_role, selected_team[0][1], "Leader")
    
    # Create agents
    agents = {}
    leader = None
    
    for role, expertise, hierarchy in selected_team:
        is_leader = hierarchy == "Leader"
        
        agent = ModularAgent(
            role_type=role,
            use_team_leadership=is_leader,
            use_closed_loop_comm=config.USE_CLOSED_LOOP_COMM,
            use_mutual_monitoring=config.USE_MUTUAL_MONITORING,
            use_shared_mental_model=config.USE_SHARED_MENTAL_MODEL
        )
        
        agents[role] = agent
        
        if is_leader:
            leader = agent
    
    # Log recruitment
    logging.info(f"Intermediate complexity: Recruited team of {len(agents)} experts with leader '{leader_role}'")
    
    return agents, leader

def recruit_advanced_team(question: str, recruitment_pool: str) -> Tuple[Dict[str, ModularAgent], ModularAgent]:
    """
    Recruit multiple specialized teams for advanced questions.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    # Create a recruiter agent to design team structure
    recruiter = Agent(
        role="Strategic Team Designer",
        expertise_description="Designs multi-team systems for complex problem solving"
    )
    
    # Get available agent types
    agent_options = config.RECRUITMENT_POOLS.get(recruitment_pool, config.RECRUITMENT_POOLS["general"])
    agent_options_text = "\n".join([f"{i+1}. {agent}" for i, agent in enumerate(agent_options)])
    
    # Create prompt for multi-team design
    design_prompt = f"""
    Given the following complex task:
    
    {question}
    
    Design 3 specialized teams to collaborate on this task:
    1. "Initial Assessment Team" (IAT) - Responsible for initial problem analysis
    2. A domain-specific team appropriate for this question
    3. "Final Decision Team" (FDT) - Responsible for integrating insights and making final recommendation
    
    Select from these available experts, and assign them to appropriate teams:
    {agent_options_text}
    
    For each team, identify a team leader who will coordinate within their team.
    Also designate ONE person as the "Chief Coordinator" who will lead the entire process.
    
    Return your design in this format:
    
    Group 1 - Initial Assessment Team (IAT)
    Member 1: [Expert Role] (Leader) - [Expertise description]
    Member 2: [Expert Role] - [Expertise description]
    Member 3: [Expert Role] - [Expertise description]
    
    Group 2 - [Domain-Specific Team Name]
    Member 1: [Expert Role] (Leader) - [Expertise description]
    Member 2: [Expert Role] - [Expertise description]
    Member 3: [Expert Role] - [Expertise description]
    
    Group 3 - Final Decision Team (FDT)
    Member 1: [Expert Role] (Chief Coordinator) - [Expertise description]
    Member 2: [Expert Role] - [Expertise description]
    Member 3: [Expert Role] - [Expertise description]
    """
    
    response = recruiter.chat(design_prompt)
    
    # Parse multi-team structure
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
            
            is_leader = '(Leader)' in parts or '(Chief Coordinator)' in parts
            is_chief = '(Chief Coordinator)' in parts
            
            # Remove leader/coordinator markers
            parts = parts.replace('(Leader)', '').replace('(Chief Coordinator)', '').strip()
            
            # Split role and expertise
            if ' - ' in parts:
                role, expertise = parts.split(' - ', 1)
            else:
                role = parts
                expertise = "General expertise in this domain"
            
            current_team['members'].append({
                'role': role.strip(),
                'expertise': expertise.strip(),
                'is_leader': is_leader,
                'is_chief': is_chief
            })
            
            if is_chief:
                chief_coordinator = role.strip()
    
    # Add the last team
    if current_team:
        teams.append(current_team)
    
    # If no teams parsed, create default team structure
    if not teams:
        teams = [
            {
                'name': 'Initial Assessment Team (IAT)',
                'members': [
                    {'role': 'Critical Analyst', 'expertise': 'Analytical rigor', 'is_leader': True, 'is_chief': False},
                    {'role': 'Systems Thinker', 'expertise': 'How components interact', 'is_leader': False, 'is_chief': False},
                    {'role': 'Domain Expert', 'expertise': 'Specialized knowledge', 'is_leader': False, 'is_chief': False}
                ]
            },
            {
                'name': 'Domain Analysis Team',
                'members': [
                    {'role': 'Domain Expert', 'expertise': 'Specialized knowledge', 'is_leader': True, 'is_chief': False},
                    {'role': 'Creative Strategist', 'expertise': 'Innovative perspectives', 'is_leader': False, 'is_chief': False},
                    {'role': 'Data Specialist', 'expertise': 'Quantitative information', 'is_leader': False, 'is_chief': False}
                ]
            },
            {
                'name': 'Final Decision Team (FDT)',
                'members': [
                    {'role': 'Process Facilitator', 'expertise': 'Methodical evaluation', 'is_leader': True, 'is_chief': True},
                    {'role': 'Critical Analyst', 'expertise': 'Analytical rigor', 'is_leader': False, 'is_chief': False},
                    {'role': 'Risk Assessor', 'expertise': 'Potential problems', 'is_leader': False, 'is_chief': False}
                ]
            }
        ]
        chief_coordinator = 'Process Facilitator'
    
    # Create all agents
    agents = {}
    leader = None
    
    # Use a prefix to make roles unique across teams
    for team_idx, team in enumerate(teams):
        team_prefix = f"{team_idx+1}_{team['name'].split('(')[0].strip()}_"
        
        for member in team['members']:
            role = member['role']
            expertise = member['expertise']
            is_leader = member['is_leader']
            is_chief = member['is_chief']
            
            # Create unique role identifier
            unique_role = f"{team_prefix}{role}"
            
            # Create agent
            agent = ModularAgent(
                role_type=unique_role,  # Use unique identifier as role type
                use_team_leadership=is_leader,  # Enable leadership for team leaders
                use_closed_loop_comm=config.USE_CLOSED_LOOP_COMM,
                use_mutual_monitoring=config.USE_MUTUAL_MONITORING,
                use_shared_mental_model=config.USE_SHARED_MENTAL_MODEL
            )
            
            # Add to agents dictionary
            agents[unique_role] = agent
            
            # Track chief coordinator as overall leader
            if is_chief:
                leader = agent
    
    # If no chief coordinator found, use first team leader
    if not leader:
        for role, agent in agents.items():
            if agent.can_lead:
                leader = agent
                break
    
    # Log recruitment
    logging.info(f"Advanced complexity: Recruited {len(agents)} experts in {len(teams)} teams")
    logging.info(f"Chief Coordinator: {chief_coordinator if chief_coordinator else 'Not specified'}")
    
    return agents, leader