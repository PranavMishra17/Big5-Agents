#!/usr/bin/env python3
"""
Test SLM models with a couple of sample dataset questions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from simulator import AgentSystemSimulator
import config
from utils.token_counter import get_token_counter, reset_global_counter

def create_sample_medqa_questions():
    """Create a couple of sample MedQA-style questions for testing."""
    return [
        {
            "question": "A 45-year-old man presents with acute chest pain that started 2 hours ago. The pain is crushing, substernal, and radiates to the left arm. He has a history of hypertension and smoking. What is the most likely diagnosis?",
            "options": {
                "A": "Gastroesophageal reflux disease",
                "B": "Acute myocardial infarction", 
                "C": "Pulmonary embolism",
                "D": "Pneumothorax"
            },
            "answer": "B",
            "subject": "Cardiology",
            "topic": "Acute coronary syndrome"
        },
        {
            "question": "A 28-year-old woman presents with fever, cough, and shortness of breath. Chest X-ray shows bilateral infiltrates. She recently returned from travel. What is the most appropriate initial treatment?",
            "options": {
                "A": "Azithromycin",
                "B": "Ceftriaxone and azithromycin",
                "C": "Oseltamivir",
                "D": "Supportive care only"
            },
            "answer": "B", 
            "subject": "Pulmonology",
            "topic": "Community-acquired pneumonia"
        }
    ]

def format_question_for_agents(question_data):
    """Format the question data for agent processing."""
    formatted_question = f"""Medical Question: {question_data['question']}

Answer Options:
A. {question_data['options']['A']}
B. {question_data['options']['B']} 
C. {question_data['options']['C']}
D. {question_data['options']['D']}

Please select the most appropriate answer and provide your reasoning."""
    
    return formatted_question

def test_slm_with_medical_questions():
    """Test SLM models with medical questions using the simulator."""
    print("=== Testing SLM Models with Medical Dataset Questions ===\n")
    
    # Reset token counter
    reset_global_counter()
    
    # Get sample questions
    sample_questions = create_sample_medqa_questions()
    
    # Test each question
    for i, question_data in enumerate(sample_questions, 1):
        print(f"--- Question {i} ---")
        print(f"Subject: {question_data['subject']} | Topic: {question_data['topic']}")
        print(f"Question: {question_data['question']}")
        print(f"Correct Answer: {question_data['answer']} - {question_data['options'][question_data['answer']]}")
        print()
        
        # Format question for agents
        formatted_question = format_question_for_agents(question_data)
        
        # Create task configuration
        task_config = {
            "name": f"Medical Question {i}",
            "description": formatted_question,
            "type": "mcq",
            "options": [
                f"A. {question_data['options']['A']}",
                f"B. {question_data['options']['B']}",
                f"C. {question_data['options']['C']}",
                f"D. {question_data['options']['D']}"
            ],
            "expected_output_format": "Single letter selection with rationale"
        }
        
        try:
            # Initialize simulator with SLM models
            simulator = AgentSystemSimulator(
                use_team_leadership=True,
                use_closed_loop_comm=False,
                use_mutual_monitoring=True,
                use_shared_mental_model=True,
                use_team_orientation=True,
                use_mutual_trust=False,
                use_agent_recruitment=True
            )
            
            # Set tracking context
            question_id = f"slm_test_q{i}"
            simulator.set_tracking_context(question_id=question_id, simulation_id="slm_test_2024")
            
            print("Running simulation with SLM models...")
            start_time = datetime.now()
            
            # Run the simulation
            result = simulator.simulate_isolated(task_config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"Simulation completed in {duration:.1f}s")
            print(f"Final answer: {result.get('final_decision', 'No decision')}")
            print(f"Agent responses: {len(result.get('agent_responses', []))} agents participated")
            
            # Show individual agent responses
            if 'agent_responses' in result:
                print("\n--- Agent Responses ---")
                for j, response in enumerate(result['agent_responses'], 1):
                    agent_role = response.get('agent_role', f'Agent {j}')
                    agent_answer = response.get('response', {})
                    if isinstance(agent_answer, dict) and 'answer' in agent_answer:
                        answer = agent_answer['answer']
                        confidence = agent_answer.get('confidence', 'Unknown')
                        print(f"{agent_role}: {answer} (Confidence: {confidence})")
                    else:
                        print(f"{agent_role}: {str(agent_answer)[:100]}...")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            print("-" * 60)
    
    # Show token usage summary
    token_counter = get_token_counter()
    usage_summary = token_counter.get_usage_summary()
    print(f"\n=== Token Usage Summary ===")
    print(f"Total API calls: {usage_summary.get('total_calls', 0)}")
    print(f"Total input tokens: {usage_summary.get('total_input_tokens', 0)}")
    print(f"Total output tokens: {usage_summary.get('total_output_tokens', 0)}")
    print(f"Total tokens: {usage_summary.get('total_tokens', 0)}")
    
    print("\n=== SLM Dataset Test Complete ===")

if __name__ == "__main__":
    test_slm_with_medical_questions()