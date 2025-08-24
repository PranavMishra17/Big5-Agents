#!/usr/bin/env python3
"""
Test script for Vertex AI SLM integration.
Tests basic functionality of the updated agent system with Vertex AI models.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from components.agent import Agent

def test_vertex_ai_configuration():
    """Test that Vertex AI configuration is properly loaded."""
    print("=== Testing Vertex AI Configuration ===")
    
    print(f"Vertex AI deployments: {len(config.VERTEX_AI_DEPLOYMENTS)}")
    for deployment in config.VERTEX_AI_DEPLOYMENTS:
        print(f"  - {deployment['name']}: {deployment['model']} (Endpoint: {deployment['endpoint_id']})")
    
    print(f"Question parallel processing: {config.ENABLE_QUESTION_PARALLEL}")
    print(f"Max parallel questions: {config.MAX_PARALLEL_QUESTIONS}")
    
    # Test deployment selection
    try:
        deployment_0 = config.get_deployment_for_agent(0)
        deployment_1 = config.get_deployment_for_agent(1)
        print(f"Agent 0 deployment: {deployment_0['name']}")
        print(f"Agent 1 deployment: {deployment_1['name']}")
        print("SUCCESS: Deployment selection working")
    except Exception as e:
        print(f"ERROR: Deployment selection failed: {e}")
    
    return len(config.VERTEX_AI_DEPLOYMENTS) > 0

def test_agent_initialization():
    """Test that agents can be initialized with Vertex AI deployments."""
    print("\n=== Testing Agent Initialization ===")
    
    try:
        # Test with MedGemma model
        medgemma_agent = Agent(
            role="Medical Specialist",
            expertise_description="Medical diagnosis expert using MedGemma-4B",
            agent_index=0
        )
        print(f"SUCCESS: MedGemma agent initialized: {medgemma_agent.deployment_type}")
        print(f"  Model: {medgemma_agent.model}")
        print(f"  Endpoint: {medgemma_agent.endpoint.resource_name}")
        
        # Test with Gemma 3 model
        gemma3_agent = Agent(
            role="General Reasoner",
            expertise_description="General reasoning specialist using Gemma-3-12B",
            agent_index=1
        )
        print(f"SUCCESS: Gemma-3 agent initialized: {gemma3_agent.deployment_type}")
        print(f"  Model: {gemma3_agent.model}")
        print(f"  Endpoint: {gemma3_agent.endpoint.resource_name}")
        
        return True, medgemma_agent, gemma3_agent
        
    except Exception as e:
        print(f"ERROR: Agent initialization failed: {e}")
        return False, None, None

def test_basic_chat(agent):
    """Test basic chat functionality."""
    print("\n=== Testing Basic Chat Functionality ===")
    
    try:
        # Simple medical question
        test_message = "What are the common causes of chest pain?"
        print(f"Test question: {test_message}")
        
        # Make the chat call
        response = agent.chat(test_message)
        
        print(f"SUCCESS: Chat successful")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Chat failed: {e}")
        return False

def main():
    """Main test function."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print("Starting Vertex AI SLM Integration Tests\n")
    
    # Test 1: Configuration
    config_ok = test_vertex_ai_configuration()
    if not config_ok:
        print("ERROR: Configuration test failed. Stopping tests.")
        return
    
    # Test 2: Agent initialization
    init_ok, medgemma_agent, gemma3_agent = test_agent_initialization()
    if not init_ok:
        print("ERROR: Agent initialization test failed. Stopping tests.")
        return
    
    # Test 3: Basic chat with MedGemma
    if medgemma_agent:
        print("\n--- Testing MedGemma Chat ---")
        medgemma_chat_ok = test_basic_chat(medgemma_agent)
    
    # Test 4: Basic chat with Gemma-3
    if gemma3_agent:
        print("\n--- Testing Gemma-3 Chat ---")
        gemma3_chat_ok = test_basic_chat(gemma3_agent)
    
    print("\n=== Test Summary ===")
    print(f"Configuration: {'SUCCESS' if config_ok else 'ERROR'}")
    print(f"Agent Initialization: {'SUCCESS' if init_ok else 'ERROR'}")
    if 'medgemma_chat_ok' in locals():
        print(f"MedGemma Chat: {'SUCCESS' if medgemma_chat_ok else 'ERROR'}")
    if 'gemma3_chat_ok' in locals():
        print(f"Gemma-3 Chat: {'SUCCESS' if gemma3_chat_ok else 'ERROR'}")
    
    print("\nVertex AI SLM integration testing complete!")

if __name__ == "__main__":
    main()