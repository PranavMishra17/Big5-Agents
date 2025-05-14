"""
Modular agent implementation for the agent system.
"""

from typing import List, Dict, Any, Optional
import random
import logging

from components.agent import Agent
import config

from typing import Tuple

from utils.prompts import LEADERSHIP_PROMPTS, TASK_ANALYSIS_PROMPTS


class ModularAgent(Agent):
    """
    Modular agent with specialization capabilities.
    This agent can specialize in different roles and optionally take leadership positions.
    """
    
    def __init__(self, 
                 role_type: str,
                 use_team_leadership: bool = False,
                 use_closed_loop_comm: bool = False,
                 use_mutual_monitoring: bool = False,
                 use_shared_mental_model: bool = False,
                    use_team_orientation: bool = False,
                    use_mutual_trust: bool = False,
                    n_max: int = 5
                 
                 ):
        """
        Initialize a team member with specific role expertise.
        
        Args:
            role_type: The specific role type (from config.AGENT_ROLES)
            use_team_leadership: Whether this agent can take leadership responsibilities
            use_closed_loop_comm: Whether to use closed-loop communication
            use_mutual_monitoring: Whether to use mutual performance monitoring
            use_shared_mental_model: Whether to use shared mental models
        """
        # Set role and expertise based on role type
        if role_type in config.AGENT_ROLES:
            self.role_type = role_type
            expertise = config.AGENT_ROLES[role_type]
        else:
            self.role_type = "General Agent"
            expertise = "General team member with balanced knowledge"
        
        # Initialize the base agent
        super().__init__(
            role=role_type,
            expertise_description=expertise,
            use_team_leadership=use_team_leadership,
            use_closed_loop_comm=use_closed_loop_comm,
            use_mutual_monitoring=use_mutual_monitoring,
            use_shared_mental_model=use_shared_mental_model,
            use_team_orientation=use_team_orientation,
            use_mutual_trust=use_mutual_trust,
            n_max=n_max
        )
        
        # Track whether this agent has leadership capabilities
        self.can_lead = use_team_leadership
        
        # Initialize specialized knowledge based on role
        self._initialize_specialized_knowledge()
        
        # Create shared knowledge repository that can be accessed by other agents
        self.shared_knowledge = {}
        
        # Additional attributes for decision making
        self.confidence = 0.0
        self.preference = None
        self.preference_ranking = []
    
    def _initialize_specialized_knowledge(self):
        """Initialize knowledge specific to the agent's specialization."""
        # Initialize basic task knowledge
        self.add_to_knowledge_base("task", config.TASK)
        
        # Role-specific initialization
        if self.role_type == "Critical Analyst":
            self._initialize_critical_analyst_knowledge()
        elif self.role_type == "Domain Expert":
            self._initialize_domain_expert_knowledge()
        elif self.role_type == "Creative Strategist":
            self._initialize_creative_strategist_knowledge()
        elif self.role_type == "Process Facilitator":
            self._initialize_process_facilitator_knowledge()
        else:
            # General knowledge for non-specialized agents
            self._initialize_general_knowledge()
            
        # Add ground truth if available (for evaluation purposes)
        # if "ground_truth" in config.TASK and "rationale" in config.TASK: self.add_to_knowledge_base("ground_truth", { #"answer": config.TASK["ground_truth"], #"rationale": config.TASK["rationale"]})
            
    
    def _initialize_critical_analyst_knowledge(self):
        """Initialize knowledge for the Critical Analyst role."""
        self.add_to_knowledge_base("critical_thinking_principles", {
            "evidence_evaluation": "Assessing quality, relevance, and sufficiency of evidence",
            "logical_reasoning": "Identifying logical fallacies and ensuring sound argumentative structure",
            "probability_assessment": "Evaluating likelihoods and considering base rates",
            "bias_identification": "Recognizing potential biases in reasoning",
            "falsification": "Actively seeking evidence that could disprove assumptions"
        })
        
        self.add_to_knowledge_base("analytical_methods", {
            "decomposition": "Breaking complex problems into manageable components",
            "cross_examination": "Testing claims by posing critical questions",
            "counterfactual_thinking": "Considering alternative scenarios and explanations",
            "sensitivity_analysis": "Testing how robust conclusions are to changes in assumptions",
            "bayesian_updating": "Adjusting beliefs based on new evidence"
        })
    
    def _initialize_domain_expert_knowledge(self):
        """Initialize knowledge for the Domain Expert role."""
        task_domain = config.TASK["name"].lower()
        
        # Customize domain knowledge based on task
        if "lunar" in task_domain or "space" in task_domain or "nasa" in task_domain:
            self._initialize_space_knowledge()
        elif "climate" in task_domain or "environment" in task_domain:
            self._initialize_climate_knowledge()
        elif "business" in task_domain or "market" in task_domain or "strategy" in task_domain:
            self._initialize_business_knowledge()
        elif "medical" in task_domain or "health" in task_domain:
            self._initialize_medical_knowledge()
        else:
            # General domain expertise approach
            self.add_to_knowledge_base("domain_expertise_approach", {
                "fact_based": "Relying on established facts and principles in the domain",
                "practical_experience": "Drawing on knowledge of real-world applications",
                "technical_precision": "Ensuring accuracy in domain-specific terminology and concepts",
                "historical_context": "Understanding how knowledge in the domain has evolved",
                "current_trends": "Awareness of recent developments and future directions"
            })
    
    def _initialize_space_knowledge(self):
        """Initialize knowledge about space and lunar environments."""
        self.add_to_knowledge_base("lunar_environment", {
            "atmosphere": "No atmosphere, complete vacuum conditions which affects heat distribution, sound transmission, and protection from radiation",
            "temperature": "Extreme variations (+250°F in direct sunlight, -250°F in shadow) with rapid shifts due to lack of atmospheric insulation",
            "gravity": "1/6 of Earth's gravity (0.166g), requiring less energy for movement but also less stability",
            "radiation": "No atmospheric or magnetic field protection from solar radiation, cosmic rays, and solar flares",
            "day_length": "14 Earth days of daylight followed by 14 Earth days of darkness due to lunar rotation",
            "terrain": "Uneven surfaces, craters, regolith (fine dust) that clings electrostatically to equipment and can damage seals",
            "visibility": "High contrast between light and shadow areas, causing visual perception difficulties"
        })
        
        self.add_to_knowledge_base("survival_principles", {
            "oxygen": "Absolutely critical; no natural oxygen on lunar surface; survival impossible beyond minutes without it",
            "water": "Critical for hydration, cooling, and preventing rapid body fluid loss in vacuum environment",
            "temperature_regulation": "Essential to manage temperature extremes from solar radiation and vacuum conditions",
            "navigation": "Stellar navigation most reliable; no magnetic field for compass; landmarks and maps crucial",
            "communication": "No atmosphere to carry sound; radio or visual signals required for distance communication",
            "radiation_protection": "Necessary to prevent acute radiation syndrome and long-term cell damage",
            "movement_efficiency": "Important to minimize oxygen consumption and avoid exhaustion in spacesuits"
        })
    
    def _initialize_climate_knowledge(self):
        """Initialize knowledge about climate science."""
        self.add_to_knowledge_base("climate_science", {
            "greenhouse_gases": "CO2, methane, nitrous oxide, water vapor, and fluorinated gases trap heat in atmosphere",
            "carbon_cycle": "Natural exchange of carbon between atmosphere, oceans, soil, and living organisms",
            "radiative_forcing": "Change in energy flux caused by drivers of climate change",
            "climate_sensitivity": "Temperature response to doubling of atmospheric CO2",
            "tipping_points": "Thresholds that, when exceeded, lead to large and often irreversible changes",
            "mitigation": "Efforts to reduce or prevent greenhouse gas emissions",
            "adaptation": "Adjusting to actual or expected climate effects"
        })
    
    def _initialize_business_knowledge(self):
        """Initialize knowledge about business and market strategy."""
        self.add_to_knowledge_base("business_strategy", {
            "market_analysis": "Evaluation of market size, trends, segments, and competitive landscape",
            "value_proposition": "Unique benefits a product or service offers to customers",
            "competitive_advantage": "Attributes that allow outperforming competitors",
            "business_models": "How an organization creates, delivers, and captures value",
            "go_to_market": "Strategy for reaching target customers and achieving competitive advantage",
            "scaling": "Growing business operations in a sustainable way",
            "risk_management": "Identifying, assessing, and controlling threats to capital and earnings"
        })
    
    def _initialize_medical_knowledge(self):
        """Initialize knowledge about medical and health topics."""
        self.add_to_knowledge_base("medical_science", {
            "diagnostic_process": "Systematic approach to determine the cause of symptoms or conditions",
            "evidence_based_medicine": "Using best available evidence for clinical decision making",
            "risk_factors": "Characteristics associated with increased likelihood of disease or condition",
            "treatment_approaches": "Therapeutic interventions including medications, procedures, lifestyle changes",
            "preventive_medicine": "Measures taken to prevent disease rather than treating symptoms",
            "public_health": "Protecting and improving health of populations rather than individuals",
            "patient_outcomes": "Results of healthcare interventions from patient perspective"
        })
    
    def _initialize_creative_strategist_knowledge(self):
        """Initialize knowledge for the Creative Strategist role."""
        self.add_to_knowledge_base("creative_thinking_principles", {
            "divergent_thinking": "Generating multiple diverse possibilities and solutions",
            "associative_thinking": "Connecting seemingly unrelated ideas and concepts",
            "perspective_shifting": "Viewing problems from multiple different angles",
            "constraint_reimagining": "Turning limitations into opportunities",
            "integrative_thinking": "Synthesizing opposing ideas into novel solutions"
        })
        
        self.add_to_knowledge_base("innovation_strategies", {
            "first_principles": "Deconstructing problems to their fundamental elements",
            "analogical_reasoning": "Transferring solutions from one domain to another",
            "scenario_planning": "Envisioning multiple possible futures to inform decisions",
            "disruptive_thinking": "Challenging established norms and assumptions",
            "rapid_prototyping": "Quickly testing ideas to gather feedback and iterate"
        })
    
    def _initialize_process_facilitator_knowledge(self):
        """Initialize knowledge for the Process Facilitator role."""
        self.add_to_knowledge_base("decision_frameworks", {
            "decision_matrix": "Evaluating options against multiple criteria",
            "pros_cons_analysis": "Weighing advantages and disadvantages of each option",
            "forced_ranking": "Comparing each option directly against others",
            "value_focused_thinking": "Identifying values first, then finding options that satisfy them",
            "devil's_advocate": "Critically challenging each option to test robustness"
        })
        
        self.add_to_knowledge_base("collaboration_techniques", {
            "structured_discussion": "Organizing conversation to ensure all voices are heard",
            "consensus_building": "Working toward solutions acceptable to all team members",
            "idea_aggregation": "Combining individual contributions into collective output",
            "conflict_resolution": "Addressing disagreements constructively to improve decisions",
            "progress_tracking": "Monitoring advancement toward decision milestones"
        })
    
    def _initialize_general_knowledge(self):
        """Initialize general knowledge for non-specialized agents."""
        self.add_to_knowledge_base("general_reasoning", {
            "critical_thinking": "Evaluating information objectively and making reasoned judgments",
            "systems_thinking": "Understanding how components interact within a larger system",
            "probabilistic_reasoning": "Making judgments under uncertainty",
            "multi-perspective_analysis": "Considering problems from different viewpoints",
            "practical_wisdom": "Applying experience and knowledge to real-world situations"
        })
    
    def analyze_task(self) -> str:
        """
        Analyze the task based on agent's specialized knowledge.
        
        Returns:
            The agent's analysis
        """
        task_type = config.TASK["type"]
        
        if task_type == "ranking":
            return self._analyze_ranking_task()
        elif task_type == "mcq":
            return self._analyze_mcq_task()
        else:
            return self._analyze_general_task()
    
    def _analyze_ranking_task(self) -> str:
        """Analyze a ranking task."""
        prompt = TASK_ANALYSIS_PROMPTS["ranking_task"].format(
            role=self.role,
            task_description=config.TASK['description'],
            items_to_rank=', '.join(config.TASK['options']),
            num_items=len(config.TASK['options'])
        )
        
        return self.chat(prompt)
    

    def _analyze_mcq_task(self) -> str:
        """Analyze a multiple-choice task."""
        prompt = TASK_ANALYSIS_PROMPTS["mcq_task"].format(
            role=self.role,
            task_description=config.TASK['description'],
            options=chr(10).join(config.TASK['options'])
        )
        
        return self.chat(prompt)   
    

    def _analyze_general_task(self) -> str:
        """Analyze a general (open-ended, estimation, etc.) task."""
        prompt = TASK_ANALYSIS_PROMPTS["general_task"].format(
            role=self.role,
            task_description=config.TASK['description']
        )
        
        return self.chat(prompt)

    def respond_to_agent(self, agent_message: str, agent_role: str) -> str:
        """Respond to another agent's message from your specialized perspective."""
        from utils.prompts import DISCUSSION_PROMPTS
        
        prompt = DISCUSSION_PROMPTS["respond_to_agent"].format(
            agent_role=agent_role,
            agent_message=agent_message,
            role=self.role
        )
        
        return self.chat(prompt)


    def leadership_action(self, action_type: str, context: str = None) -> str:
        """
        Perform a leadership action if this agent has leadership capabilities.
        
        Args:
            action_type: Type of leadership action
            context: Optional context information
            
        Returns:
            Result of the leadership action, or explanation if not a leader
        """
        if not self.can_lead:
            return f"As a {self.role} without leadership designation, I cannot perform leadership actions."
        
        if action_type == "define_task":
            prompt = LEADERSHIP_PROMPTS["define_task"].format(
                task_description=config.TASK['description'],
                context=context or ''
            )
        elif action_type == "synthesize":
            prompt = LEADERSHIP_PROMPTS["synthesize"].format(
                context=context or ''
            )
        elif action_type == "facilitate":
            prompt = LEADERSHIP_PROMPTS["facilitate"].format(
                context=context or ''
            )
        else:
            return f"Unknown leadership action: {action_type}"
        
        return self.chat(prompt)



# Modify the create_agent_team function to support recruitment
def create_agent_team(use_team_leadership=True, 
                      use_closed_loop_comm=False, 
                      use_mutual_monitoring=False,
                      use_shared_mental_model=False,
                      use_team_orientation=False,
                      use_mutual_trust=False,
                      random_leader=False,
                      use_recruitment=False,
                      question=None,
                      recruitment_method="adaptive",
                      recruitment_pool="general",
                      n_max=5) -> Tuple[Dict[str, ModularAgent], ModularAgent]:
    """
    Create a team of agents with different specializations.
    
    Args:
        use_team_leadership: Whether to use team leadership
        use_closed_loop_comm: Whether to use closed-loop communication
        use_mutual_monitoring: Whether to use mutual performance monitoring
        use_shared_mental_model: Whether to use shared mental models
        use_team_orientation: Whether to use team orientation
        use_mutual_trust: Whether to use mutual trust
        random_leader: Whether to randomly assign leadership
        use_recruitment: Whether to use dynamic agent recruitment
        question: The question or task (required if use_recruitment is True)
        recruitment_method: Method for recruitment (adaptive, fixed, basic, intermediate, advanced)
        recruitment_pool: Pool of agent roles to recruit from
        n_max: Maximum number of agents for intermediate team
        
    Returns:
        Tuple of (agents dictionary, leader agent)
    """

    # Use recruitment if enabled
    if use_recruitment and question:
        # Import here to avoid circular imports
        from components.agent_recruitment import determine_complexity, recruit_agents
        
        # Determine complexity
        complexity = determine_complexity(question, recruitment_method)
        
        # Recruit appropriate agents
        return recruit_agents(question, complexity, recruitment_pool, n_max)
    
    # Default to original implementation if recruitment not enabled
    # Determine leadership assignment
    if random_leader:
        # Randomly choose one agent to be leader
        leader_role = random.choice(list(config.AGENT_ROLES.keys()))
    elif use_team_leadership:
        # By default, make the Process Facilitator the leader if available
        leader_role = "Process Facilitator" if "Process Facilitator" in config.AGENT_ROLES else list(config.AGENT_ROLES.keys())[0]
    else:
        leader_role = None
    
    # Create all agents
    agents = {}
    leader = None
    
    for role in config.AGENT_ROLES:
        # Determine if this agent should have leadership capabilities
        is_leader = role == leader_role if leader_role else False
        
        # Create the agent
        agent = ModularAgent(
            role_type=role,
            use_team_leadership=is_leader and use_team_leadership,
            use_closed_loop_comm=use_closed_loop_comm,
            use_mutual_monitoring=use_mutual_monitoring,
            use_shared_mental_model=use_shared_mental_model,
            use_team_orientation=use_team_orientation,
            use_mutual_trust=use_mutual_trust,
            n_max=n_max
        )
        
        agents[role] = agent
        
        # Track the leader agent
        if is_leader and use_team_leadership:
            leader = agent
    
    # Share knowledge between agents
    for role1, agent1 in agents.items():
        for role2, agent2 in agents.items():
            if role1 != role2:
                agent1.share_knowledge(agent2)
    
    # Log the team configuration
    logging.info(f"Created agent team with configuration:")
    logging.info(f"  Team Leadership: {use_team_leadership} (Leader: {leader_role if leader_role else 'None'})")
    logging.info(f"  Closed-loop Communication: {use_closed_loop_comm}")
    logging.info(f"  Mutual Performance Monitoring: {use_mutual_monitoring}")
    logging.info(f"  Shared Mental Model: {use_shared_mental_model}")
    logging.info(f"  Team Orientation: {use_team_orientation}")
    logging.info(f"  Mutual Trust: {use_mutual_trust}")
    logging.info(f" n_max: {n_max}")
    logging.info(f"  Agents: {', '.join(agents.keys())}")
    
    return {"agents": agents, "leader": leader}