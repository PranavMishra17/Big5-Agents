#!/usr/bin/env python3
"""
Full integration test with actual medical questions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.agent import Agent

def test_medical_questions():
    """Test with actual medical questions."""
    print("=== Full Vertex AI SLM Integration Test ===\n")
    
    # Create agents
    medgemma_agent = Agent(
        role="Medical Specialist",
        expertise_description="Medical diagnosis expert using MedGemma-4B",
        agent_index=1  # Uses medgemma_4b_1
    )
    
    gemma3_agent = Agent(
        role="General Reasoner", 
        expertise_description="General reasoning specialist using Gemma-3-12B",
        agent_index=0  # Uses gemma3_12b_1
    )
    
    # Test questions
    test_questions = [
        "List the 5 most common causes of chest pain in adults.",
        "What are the key symptoms of pneumonia?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
    ]
    
    print("--- Testing MedGemma-4B ---")
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        try:
            response = medgemma_agent.chat(question)
            print(f"MedGemma Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n--- Testing Gemma-3-12B ---")
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        try:
            response = gemma3_agent.chat(question)
            print(f"Gemma-3 Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    test_medical_questions()