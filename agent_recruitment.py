"""
Agent recruitment module for dynamically assembling specialized agent teams.
"""

import logging
import random
from typing import Dict, List, Any, Tuple

from agent import Agent
from modular_agent import ModularAgent, create_agent_team
import config

def determine_complexity(question, method="adaptive"):
    """
    Determine the complexity of a question to decide on team structure.
    
    Args:
        question: The question or task to analyze
        method: Method for complexity determination
        
    Returns:
        Complexity level ("basic", "intermediate", or "advanced")
    """
    if method == "basic":
        return "basic"
    elif method == "intermediate":
        return "intermediate"
    elif method == "advanced":
        return "advanced"
    
    # For adaptive method, evaluate the question
    evaluator = Agent(
        role="Complexity Evaluator",
        expertise_description="analyzes tasks to determine their complexity level"
    )
    
    prompt = f"""
    Analyze the following task/question and determine its complexity level:
    
    {question}
    
    Based on your analysis, classify this as one of:
    1) basic - A single expert can handle this question (simpler questions, single domain)
    2) intermediate - Requires a team approach (moderately complex, 2-3 domains involved)
    3) advanced - Requires multiple experts with diverse knowledge (complex, interdisciplinary, novel)
    
    Consider these factors:
    - Are multiple domains of expertise needed?
    - Is there uncertainty or ambiguity requiring multiple perspectives?
    - Are there competing considerations or complex tradeoffs?
    - Does it require specialized medical knowledge?
    - Are there diagnostic complexities or rare conditions?
    - Does it involve analysis of imaging, laboratory results or complex symptom patterns?
    
    Format your response as:
    **Complexity Classification:** [number]) [complexity level]
    """
    
    response = evaluator.chat(prompt)
    
    # Extract complexity classification
    if "basic" in response.lower():
        complexity = "basic"
    elif "intermediate" in response.lower():
        complexity = "intermediate"
    elif "advanced" in response.lower():
        complexity = "advanced"
    else:
        # Default to intermediate if parsing fails
        complexity = "intermediate"
    
    # Override for medical/diagnostic questions to ensure at least intermediate complexity
    if any(term in question.lower() for term in ["diagnosis", "diagnostic", "symptom", "clinical", "patient", "disease", "disorder", "syndrome", "encephalitis", "antibody", "autoimmune"]):
        if complexity == "basic":
            complexity = "intermediate"
    
    logging.info(f"{method.capitalize()} complexity: {complexity}")
    
    return complexity

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
    Recruit a single specialized expert for basic questions, following the MDAgents approach.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    # Create a medical agent to handle basic queries
    medical_agent = Agent(
        role="Medical Agent",
        expertise_description="A specialized medical expert with comprehensive medical knowledge"
    )
    
    # Initialize the prompt
    system_prompt = "You are a helpful medical agent that answers multiple choice questions about medical knowledge."
    
    # For basic tasks, we follow the MDAgents paper by having a single specialized agent
    # with task-specific expertise rather than using our default roles
    
    # Create the specialized agent with a specific role based on question analysis
    analysis_prompt = f"""
    Based on the following medical question, what medical specialty would be most appropriate 
    to accurately answer this question? Please identify only ONE specific medical specialty 
    that is most relevant.
    
    Question: {question}
    
    Please respond with just the specialty name (e.g., 'Cardiologist', 'Neurologist', etc.)
    """
    
    # Get the appropriate specialty
    specialty_response = medical_agent.chat(analysis_prompt)
    
    # Extract specialty (simple approach - first line or first sentence)
    specialty = specialty_response.strip().split('\n')[0].split('.')[0]
    
    # Clean up any extraneous text
    if ':' in specialty:
        specialty = specialty.split(':', 1)[1].strip()
    
    # Create a domain-specific expertise description
    expertise_description = f"Specialized in {specialty} with comprehensive medical knowledge"
    
    # Create the specialized agent
    agent = ModularAgent(
        role_type=specialty, 
        use_team_leadership=True,  # Single agent is always its own leader
        use_closed_loop_comm=config.USE_CLOSED_LOOP_COMM,
        use_mutual_monitoring=config.USE_MUTUAL_MONITORING,
        use_shared_mental_model=config.USE_SHARED_MENTAL_MODEL
    )
    
    # Create the agents dictionary with our single specialized agent
    agents = {specialty: agent}
    
    # Log recruitment
    logging.info(f"Basic complexity: Recruited single expert '{specialty}'")
    
    return agents, agent


def recruit_intermediate_team(question: str, recruitment_pool: str) -> Tuple[Dict[str, ModularAgent], ModularAgent]:
    """
    Recruit a team of specialists with hierarchical relationships, following MDAgents approach.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    # Create a recruiter agent
    recruiter = Agent(
        role="Recruiter",
        expertise_description="Assembles teams of medical experts for collaborative problem-solving"
    )
    
    # Determine number of agents to recruit (5 in MDAgents paper)
    num_agents = 5
    
    # Create prompt for team selection following MDAgents approach
    selection_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query.
    
IMPORTANT: Select experts with DISTINCT and NON-OVERLAPPING specialties that are directly relevant to the medical question. Each expert should bring a unique perspective or knowledge domain.

Question: {question}

You can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?

Also, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.

For example, if you want to recruit five experts, your answer can be like:
1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent
2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist
3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent
4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent
5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent

Please answer in above format, and do not include your reason.
"""
    
    response = recruiter.chat(selection_prompt)
    
    # Parse selected team - similar to MDAgents
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    selected_team = []
    leader_role = None
    hierarchies = {}
    
    for line in lines:
        if " - Hierarchy: " not in line:
            continue
            
        # Extract role, expertise, and hierarchy
        parts = line.split(" - Hierarchy: ")
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
        if "Independent" in hierarchy:
            hierarchies[role] = "Independent"
        else:
            # Check if this role appears as the superior in any hierarchy
            for other_role in hierarchies:
                if f"{role} > " in hierarchy:
                    # This role is superior to others
                    if leader_role is None:
                        leader_role = role
        
        selected_team.append((role, expertise, hierarchies.get(role, "Independent")))
        
        # Stop when we have enough agents
        if len(selected_team) >= num_agents:
            break
    
    # If no team members found, create a default team
    if not selected_team:
        selected_team = [
            ("Internist", "General medical knowledge and diagnosis", "Leader"),
            ("Cardiologist", "Heart and cardiovascular system", "Independent"),
            ("Neurologist", "Brain and nervous system disorders", "Independent"),
            ("Pathologist", "Disease diagnosis through laboratory tests", "Independent"),
            ("Radiologist", "Medical imaging interpretation", "Independent")
        ]
        leader_role = "Internist"
    
    # If no leader designated, choose the first as leader
    if not leader_role:
        leader_role = selected_team[0][0]
        selected_team[0] = (leader_role, selected_team[0][1], "Leader")
    
    # Create agents
    agents = {}
    leader = None
    
    for role, expertise, hierarchy in selected_team:
        is_leader = hierarchy == "Leader" or role == leader_role
        
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
    Recruit multiple specialized teams for advanced questions, following MDAgents approach.
    
    Args:
        question: The question or task
        recruitment_pool: The pool of agent types
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """
    # Create a strategic team designer
    recruiter = Agent(
        role="Strategic Team Designer",
        expertise_description="Designs multi-team systems for complex medical problem solving"
    )
    
    # Create prompt for multi-team design following MDAgents MDT approach
    design_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer.

Question: {question}

You should organize 3 MDTs with different specialties or purposes and each MDT should have 3 clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.

For example, the following can be an example answer:
Group 1 - Initial Assessment Team (IAT)
Member 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.
Member 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.
Member 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.

Group 2 - Diagnostic Evidence Team (DET)
Member 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.
Member 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.
Member 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.

Group 3 - Final Review and Decision Team (FRDT)
Member 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.
Member 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.
Member 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.

Above is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.
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
    
    # If no teams parsed, create default team structure (similar to MDAgents)
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
    
    # Create all agents
    agents = {}
    leader = None
    
    # Use a prefix to make roles unique across teams
    for team_idx, team in enumerate(teams):
        team_prefix = f"{team_idx+1}_{team['name'].split('(')[0].strip()}_"
        
        # Update in recruit_advanced_team function - modify when creating agents
        for member in team['members']:
            role = member['role']
            expertise = member['expertise']
            is_leader = member['is_leader']
            
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
