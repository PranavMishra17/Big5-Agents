"""
Enhanced prompts.py with proper round separation and final decision prompts.
"""

# Agent System Prompts
AGENT_SYSTEM_PROMPTS = {
    "base": """You are a {role} who {expertise_description}. Your job is to collaborate with other medical experts in a team.
    
    You are working on the following task: {task_name}
    
    {task_description}
    
    This is a {task_type} task, and your output should be in the format: {expected_output_format}
    """,
}

# Team Leadership Prompts
LEADERSHIP_PROMPTS = {
    "team_leadership": """
    As part of this team, you should demonstrate effective team leadership by:
    1. Facilitating team problem solving
    2. Providing performance expectations and acceptable interaction patterns
    3. Synchronizing and combining individual team member contributions
    4. Seeking and evaluating information that affects team functioning
    5. Clarifying team member roles
    6. Engaging in preparatory discussions and feedback sessions with the team
    """,
    
    "define_task": """
    As the designated leader for this task, define the overall approach for solving:
    
    {task_description}
    
    Break this down into clear steps, specifying:
    1. The objective of each subtask
    2. The sequence in which they should be completed
    3. How to evaluate successful completion
    
    Provide clear, specific guidance that will guide the team through this process.
    
    Additional context: {context}
    """,
    
    "synthesize": """
    As the designated leader, synthesize the team's perspectives into a consensus solution.
    
    Context information: {context}
    
    Create a final solution that:
    1. Incorporates the key insights from each team member
    2. Balances different perspectives from team members
    3. Provides clear reasoning for the final decision

    You MUST begin your response with "ANSWER: X" (replace X with the letter of your chosen option A, B, C, or D).

    Present your final solution with comprehensive justification.
    Ensure all required elements from the task are addressed.
    """,
    
    "synthesize_multi_choice": """
    As the designated leader, synthesize the team's perspectives into a consensus solution for this multi-choice question.
    
    Context information: {context}
    
    Create a final solution that:
    1. Incorporates the key insights from each team member
    2. Balances different perspectives from team members
    3. Provides clear reasoning for the final selection of multiple options

    You MUST begin your response with "ANSWERS: X,Y,Z" (replace X,Y,Z with the letters of ALL correct options, e.g., "ANSWERS: A,C" or "ANSWERS: B,D").

    Present your final solution with comprehensive justification for each selected option.
    Remember: This is a multi-choice question where multiple answers may be correct.
    """,
    
    "synthesize_yes_no_maybe": """
    As the designated leader, synthesize the team's perspectives into a consensus answer for this research question.
    
    Context information: {context}
    
    Create a final answer that:
    1. Incorporates the key scientific evidence from each team member
    2. Balances different perspectives from team members
    3. Provides clear scientific reasoning for the conclusion

    You MUST begin your response with "ANSWER: X" (replace X with yes, no, or maybe).

    Present your final answer with comprehensive scientific justification based on the abstract context.
    Ensure your reasoning is grounded in the evidence presented.
    """,
    
    "facilitate": """
    As the designated leader, facilitate progress on our current discussion.
    
    Current context: {context}
    
    Please:
    1. Identify areas of agreement and disagreement
    2. Propose next steps to move the discussion forward
    3. Ask specific questions to gather needed information
    4. Summarize key insights so far
    
    Your goal is to help the team make progress rather than advocate for a specific position.
    """
}

# Closed-Loop Communication Prompts
COMMUNICATION_PROMPTS = {
    "closed_loop": """
    When communicating with your teammates, you should use closed-loop communication:
    1. When you send information, make it clear and specific
    2. When you receive information, acknowledge receipt and confirm understanding
    3. When your sent information is acknowledged, verify that it was understood correctly
    
    This three-step process ensures that critical information is properly exchanged.
    """,
    
    "receiver_acknowledgment": """
    You have received the following message from the {sender_role}:
    
    "{sender_message}"
    
    Following closed-loop communication protocol:
    1. Acknowledge that you have received this message
    2. Confirm your understanding by restating the key points in your own words
    3. Then provide your response to the content
    
    Begin your response with an acknowledgment and confirmation.
    """,
    
    "sender_verification": """
    You sent the following message to the {receiver_role}:
    
    "{sent_message}"
    
    They responded with:
    
    "{response_message}"
    
    Following closed-loop communication protocol:
    1. Verify whether they understood your message correctly
    2. Clarify any misunderstandings if necessary
    3. Then continue the conversation based on their response
    
    Begin your response with verification of their understanding.
    """
}

# Mutual Monitoring Prompts
MONITORING_PROMPTS = {
    "mutual_monitoring": """
    You should engage in mutual performance monitoring by:
    1. Tracking the performance of your teammates
    2. Checking for errors or omissions in their reasoning
    3. Providing constructive feedback when you identify issues
    4. Ensuring the overall quality of the team's work
    
    Do this respectfully and with the goal of improving the team's decision-making.
    """
}

# Shared Mental Model Prompts
MENTAL_MODEL_PROMPTS = {
    "shared_mental_model": """
    You should actively contribute to the team's shared mental model by:
    1. Explicitly stating your understanding of key concepts
    2. Checking alignment on how the team interprets the task
    3. Establishing shared terminology and frameworks
    4. Clarifying your reasoning process so others can follow it
    
    This helps ensure all team members are aligned in their understanding.
    """,
    
    "team_mental_model": """Collaboratively and accurately answer the following question, considering all provided options: {task_description}. 
    
    Maintain shared mental models to support collaboration:

    Team-related mental models:
    1. Each expert understands the other's specialty and defers appropriately. Example: The pathologist handles histo slides, labs, and biopsy findings; the internist interprets clinical presentation and management strategies.
    2. Use clear, structured communication with confirmation and clarifying questions.
    3. Agree on how to approach each question: e.g., one summarizes the case, the other analyzes the key findings, then they jointly choose the best answer.
    4. A shared attitude of psychological safety—both are open to being wrong and correcting each other without ego. Each expert is open to changing their professional opinion if compelling evidence or reasoning is presented by other experts. When sharing their views, clearly state their current decision and reasoning. If their opinion changes during the discussion, explicitly acknowledge this change and explain why.
    5. Efficient division of labor: e.g., one expert takes lead on clinical scenario interpretation, the other cross-checks with pathology or physiology knowledge.

    Task-specific understanding:
    Objective: {objective}

    Key evaluation criteria:
    {criteria}
    """,
    
    "individual_mental_model": """You are working on the following task: {task_description}

    Task-related mental models:
    1. Be aligned on what the question is asking: diagnosis, next best step, mechanism, prognosis, etc.
    2. Eliminate obviously wrong options and discuss remaining choices based on evidence.
    3. Maintain a shared baseline of question content scope—understand that questions test integration of disciplines, not isolated facts.
    4. Prioritize findings appropriately: e.g., which symptoms are more diagnostic, what labs carry more weight, etc.
    5. You are not allowed to consult external sources.
    6. You do not have any additional information other than what is provided in the question.
    7. Reflect on how different perspectives affect initial assessment of the question: If your answer changed based on this reflection, provide your decision and a brief explanation that includes whether your opinion has changed and why.

    Task-specific understanding:
    Objective: {objective}

    Key evaluation criteria:
    {criteria}
    """
}

# Team Orientation Prompts
ORIENTATION_PROMPTS = {
    "team_orientation": """
    You should demonstrate strong team orientation by:
    1. Taking into account alternative solutions provided by teammates and appraising their input
    2. Considering the value of teammates' perspectives even when they differ from your own
    3. Prioritizing team goals over individual recognition or achievements
    4. Actively engaging in information sharing, strategizing, and participatory goal setting
    5. Enhancing your individual performance through coordination with and utilization of input from teammates

    Remember that team orientation is not just working with others, but truly valuing and incorporating diverse perspectives.
    """,
    
    "enhanced_orientation": """
    As a team member with strong team orientation:
    
    1. Value and consider the input and perspectives of your teammates, even when they differ from your own
    2. Share your knowledge and information openly to benefit the team's understanding
    3. Prioritize team goals over individual recognition or achievement
    4. Actively participate in goal setting and strategy development
    5. Consider how your contributions can complement and enhance the work of others
    
    Remember that team success depends on the integration of diverse perspectives and the coordination of individual efforts toward collective goals.
    """
}

# Mutual Trust Prompts
TRUST_PROMPTS = {
    "mutual_trust_base": """
    You should foster and demonstrate mutual trust with your teammates by:
    1. Sharing information openly and appropriately without excessive protection or checking
    2. Being willing to admit mistakes and accept feedback from teammates
    3. Assuming teammates are acting with positive intentions for the team's benefit
    4. Recognizing and respecting the expertise and rights of all team members
    5. Being willing to make yourself vulnerable by asking for help when needed
    """,
    
    "low_trust": """
    However, be aware that the current team environment has LOW TRUST levels. This means:
    - Be more careful about sharing sensitive or potentially controversial information
    - Verify information from teammates more carefully before acting on it
    - Be more explicit about your reasoning and evidence to build credibility
    - Take extra steps to demonstrate reliability and consistency in your contributions
    - Work gradually to establish trust through small, reliable interactions
    """,
    
    "high_trust": """
    The current team environment has HIGH TRUST levels. This means:
    - You can share information with confidence it will be used appropriately
    - You can rely on teammates' input without excessive verification
    - You can feel safe to express uncertainty or vulnerability
    - You can expect teammates will protect the team's interests and your contributions
    - The team can focus fully on the task rather than monitoring each other
    """,
    
    "high_trust_environment": """
    As a team member in a high-trust environment:
    
    1. Share information openly and completely with your teammates
    2. Be willing to admit mistakes and uncertainties
    3. Accept feedback from others without defensiveness
    4. Trust others' expertise and judgment in their areas of specialization
    5. Feel safe in expressing vulnerability and asking for help when needed
    
    The team environment supports psychological safety, allowing for open communication and constructive feedback.
    """,
    
    "medium_trust_environment": """
    As a team member in a developing trust environment:
    
    1. Share important information with your teammates while considering what is appropriate
    2. Be willing to acknowledge mistakes when necessary
    3. Consider feedback from others carefully
    4. Recognize others' expertise in their specialized areas
    5. Balance self-reliance with appropriate requests for assistance
    
    The team is working to build trust through consistent and reliable interactions.
    """,
    
    "low_trust_environment": """
    As a team member in a trust-building environment:
    
    1. Share information that is necessary for team functioning
    2. Acknowledge when corrections are needed
    3. Provide and receive feedback in a constructive manner
    4. Demonstrate reliability in your area of expertise
    5. Gradually increase openness as trust develops
    
    The team needs to establish trust through consistent performance and respect for boundaries.
    """
}

# Agent Recruitment Prompts
RECRUITMENT_PROMPTS = {
    "complexity_evaluation": """
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
    """,
    
    "medical_specialty": """
    Based on the following medical question, what medical specialty would be most appropriate 
    to accurately answer this question? Please identify only ONE specific medical specialty 
    that is most relevant.
    
    Question: {question}
    
    Please respond with just the specialty name (e.g., 'Cardiologist', 'Neurologist', etc.)
    """,
    
    "team_selection": """You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query.
        
    IMPORTANT: Select experts with DISTINCT and NON-OVERLAPPING specialties that are directly relevant to the medical question. Each expert should bring a unique perspective or knowledge domain.

    Question: {question}

    You can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?

    For each expert, you MUST assign a weightage between 0.0 and 1.0 that reflects their importance to this specific question. The total of all weights should sum to 1.0.

    Also, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.

    For example, if you want to recruit five experts, your answer can be like:
    1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent - Weight: 0.2
    2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist - Weight: 0.15
    3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent - Weight: 0.25
    4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent - Weight: 0.2
    5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent - Weight: 0.2

    Please answer in above format, and do not include your reason.
    """,
    
    "mdt_design": """You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer.

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
}

# Task Analysis Prompts - Enhanced with clearer round separation
TASK_ANALYSIS_PROMPTS = {
    "ranking_task": """
    As a {role} with expertise in your domain, analyze the following ranking task INDEPENDENTLY:
    
    {task_description}
    
    Items to rank:
    {items_to_rank}
    
    IMPORTANT: You are working independently in this round. You have no knowledge of other team members or their thoughts.
    
    Based on your specialized knowledge, provide:
    1. Your analytical approach to this ranking task
    2. Key factors you'll consider in making your decisions
    3. Any specific insights your role brings to this type of task
    
    Then provide your complete ranking from 1 (most important) to {num_items} (least important).
    Present your ranking as a numbered list with brief justifications for each item's placement.
    
    You MUST provide a final ranking, but focus primarily on your reasoning process.
    """,
    
    "mcq_task": """
    As a {role} with expertise in your domain, analyze the following multiple-choice question INDEPENDENTLY:

    {task_description}

    Options:
    {options}

    IMPORTANT: You are working independently in this round. You have no knowledge of other team members or their thoughts.

    Based on your specialized knowledge:
    1. Analyze each option systematically
    2. Apply relevant principles from your area of expertise
    3. Consider the strengths and weaknesses of each option
    4. Provide your reasoning process

    You MUST provide your answer, but focus primarily on your analytical reasoning.
    Start your final answer with "ANSWER: X" (where X is the correct option letter) at the end of your analysis.
    """,
    
    "multi_choice_mcq_task": """
    As a {role} with expertise in your domain, analyze the following multi-choice question INDEPENDENTLY where MULTIPLE answers may be correct:

    {task_description}

    Options:
    {options}

    IMPORTANT: You are working independently in this round. You have no knowledge of other team members or their thoughts.
    
    IMPORTANT: This is a multi-choice question where MORE THAN ONE answer may be correct. You must select ALL correct options.

    Based on your specialized knowledge:
    1. Analyze each option systematically for correctness
    2. Apply relevant principles from your area of expertise
    3. Identify ALL options that are correct or appropriate
    4. Provide your reasoning process

    You MUST provide your answers, but focus primarily on your analytical reasoning.
    Start your final answer with "ANSWERS: X,Y,Z" (where X,Y,Z are ALL correct option letters) at the end of your analysis.
    """,
    
    "yes_no_maybe_task": """
    As a {role} with expertise in your domain, analyze the following research question INDEPENDENTLY:

    {task_description}

    IMPORTANT: You are working independently in this round. You have no knowledge of other team members or their thoughts.

    Based on the abstract context provided:
    1. Analyze the scientific evidence systematically
    2. Apply relevant principles from your area of expertise
    3. Consider whether the evidence supports, refutes, or is inconclusive regarding the research question
    4. Provide your reasoning process

    You MUST provide your answer, but focus primarily on your analytical reasoning.
    Begin your final answer with "ANSWER: X" (replace X with yes, no, or maybe) at the end of your analysis.
    """,
    
    "general_task": """
    As a {role} with expertise in your domain, analyze the following task INDEPENDENTLY:
    
    {task_description}
    
    IMPORTANT: You are working independently in this round. You have no knowledge of other team members or their thoughts.
    
    Based on your specialized knowledge:
    1. Break down the key components of this task
    2. Identify the most important factors to consider
    3. Apply relevant principles from your area of expertise
    4. Provide your reasoning process
    
    Then provide your comprehensive response to the task.
    Structure your answer clearly and provide justifications for your key points.
    """
}

# Final Decision Prompts - NEW SECTION for Round 3
FINAL_DECISION_PROMPTS = {
    "mcq_final": """
    Based on your initial analysis and the team discussion, provide your FINAL INDEPENDENT answer to this multiple-choice question.
    
    Your initial analysis:
    {initial_analysis}
    
    Team discussion insights:
    {discussion_summary}
    
    IMPORTANT: You are now making your final independent decision. Consider the insights from the discussion, but make your own judgment.
    
    Provide your final answer with reasoning.
    You MUST begin your response with "ANSWER: X" (replace X with your chosen option letter).
    Then explain your final reasoning, including how the team discussion influenced (or didn't influence) your decision.
    """,
    
    "multi_choice_final": """
    Based on your initial analysis and the team discussion, provide your FINAL INDEPENDENT answer to this multi-choice question.
    
    Your initial analysis:
    {initial_analysis}
    
    Team discussion insights:
    {discussion_summary}
    
    IMPORTANT: You are now making your final independent decision. This is a multi-choice question where multiple answers may be correct.
    Consider the insights from the discussion, but make your own judgment.
    
    Provide your final answer with reasoning.
    You MUST begin your response with "ANSWERS: X,Y,Z" (replace with ALL correct option letters, e.g., "ANSWERS: A,C").
    Then explain your final reasoning, including how the team discussion influenced (or didn't influence) your decision.
    """,
    
    "yes_no_maybe_final": """
    Based on your initial analysis and the team discussion, provide your FINAL INDEPENDENT answer to this research question.
    
    Your initial analysis:
    {initial_analysis}
    
    Team discussion insights:
    {discussion_summary}
    
    IMPORTANT: You are now making your final independent decision. Consider the insights from the discussion, but make your own judgment.
    
    Provide your final answer with reasoning.
    You MUST begin your response with "ANSWER: X" (replace X with yes, no, or maybe).
    Then explain your final reasoning, including how the team discussion influenced (or didn't influence) your decision.
    """,
    
    "ranking_final": """
    Based on your initial analysis and the team discussion, provide your FINAL INDEPENDENT ranking.
    
    Your initial analysis:
    {initial_analysis}
    
    Team discussion insights:
    {discussion_summary}
    
    IMPORTANT: You are now making your final independent decision. Consider the insights from the discussion, but make your own judgment.
    
    Provide your final ranking with reasoning.
    Present your ranking as a numbered list from 1 (most important) to N (least important).
    Then explain your final reasoning, including how the team discussion influenced (or didn't influence) your decision.
    """,
    
    "general_final": """
    Based on your initial analysis and the team discussion, provide your FINAL INDEPENDENT response.
    
    Your initial analysis:
    {initial_analysis}
    
    Team discussion insights:
    {discussion_summary}
    
    IMPORTANT: You are now making your final independent decision. Consider the insights from the discussion, but make your own judgment.
    
    Provide your final response with reasoning.
    Explain how the team discussion influenced (or didn't influence) your decision.
    """
}

# Collaborative Discussion Prompts - Enhanced for Round 2
DISCUSSION_PROMPTS = {
    "respond_to_agent": """
    Another team member ({agent_role}) has provided the following input:
    
    "{agent_message}"
    
    As a {role} with your particular expertise, respond to this message.
    Apply your specialized knowledge to this discussion about the task.
    
    If you notice any misconceptions or have additional information to add from your
    area of expertise, share that information.
    
    If you agree with their analysis, acknowledge the points you agree with and
    add any additional insights from your perspective.
    
    If you are using closed-loop communication, acknowledge the message and confirm your
    understanding before providing your response.
    """,
    
    # Remove the old collaborative_discussion prompts that had explicit answers
    # These are now handled in the final decision prompts
}

def get_adaptive_prompt(base_key: str, task_type: str, **kwargs) -> str:
    """
    Get adaptive prompt based on task type with enhanced round support.
    
    Args:
        base_key: Base prompt key (e.g., "task_analysis", "final_decision")
        task_type: Task type ("mcq", "multi_choice_mcq", "yes_no_maybe", etc.)
        **kwargs: Additional formatting arguments
        
    Returns:
        Formatted prompt appropriate for the task type and round
    """
    # Handle task analysis prompts (Round 1)
    if base_key == "task_analysis":
        if task_type == "ranking":
            return TASK_ANALYSIS_PROMPTS["ranking_task"].format(**kwargs)
        elif task_type == "mcq":
            return TASK_ANALYSIS_PROMPTS["mcq_task"].format(**kwargs)
        elif task_type == "multi_choice_mcq":
            return TASK_ANALYSIS_PROMPTS["multi_choice_mcq_task"].format(**kwargs)
        elif task_type == "yes_no_maybe":
            return TASK_ANALYSIS_PROMPTS["yes_no_maybe_task"].format(**kwargs)
        else:
            return TASK_ANALYSIS_PROMPTS["general_task"].format(**kwargs)
    
    # Handle final decision prompts (Round 3)
    elif base_key == "final_decision":
        if task_type == "ranking":
            return FINAL_DECISION_PROMPTS["ranking_final"].format(**kwargs)
        elif task_type == "mcq":
            return FINAL_DECISION_PROMPTS["mcq_final"].format(**kwargs)
        elif task_type == "multi_choice_mcq":
            return FINAL_DECISION_PROMPTS["multi_choice_final"].format(**kwargs)
        elif task_type == "yes_no_maybe":
            return FINAL_DECISION_PROMPTS["yes_no_maybe_final"].format(**kwargs)
        else:
            return FINAL_DECISION_PROMPTS["general_final"].format(**kwargs)
    
    # Handle leadership synthesis prompts
    elif base_key == "leadership_synthesis":
        if task_type == "multi_choice_mcq":
            return LEADERSHIP_PROMPTS["synthesize_multi_choice"].format(**kwargs)
        elif task_type == "yes_no_maybe":
            return LEADERSHIP_PROMPTS["synthesize_yes_no_maybe"].format(**kwargs)
        else:
            return LEADERSHIP_PROMPTS["synthesize"].format(**kwargs)
    
    # Default fallback
    else:
        return f"Unknown prompt key: {base_key}"