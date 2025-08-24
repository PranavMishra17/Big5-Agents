#!/usr/bin/env python3
"""
Direct HTTP test for dedicated Vertex AI endpoints.
"""

import requests
import json
import google.auth
from google.auth.transport.requests import Request

def test_dedicated_endpoint(endpoint_url, payload):
    """Test direct HTTP call to dedicated endpoint."""
    
    # Get authentication token
    credentials, project = google.auth.default()
    auth_req = Request()
    credentials.refresh(auth_req)
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }
    
    # Try different paths
    paths_to_try = [
        ":predict",
        "/predict", 
        "/v1:predict",
        "/v1/predict",
        "",
        ":generateContent",
        "/generateContent"
    ]
    
    for path in paths_to_try:
        try:
            url = f"https://{endpoint_url}{path}"
            print(f"Trying: {url}")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            if response.status_code != 404:
                print(f"Response: {response.text}")
                if response.status_code == 200:
                    return response.json()
                
        except Exception as e:
            print(f"Error with {path}: {e}")
    
    return None

if __name__ == "__main__":
    # Test payloads for different model formats
    payloads_to_try = [
        {
            "instances": [
                {"prompt": "What are the common causes of chest pain?"}
            ]
        },
        {
            "instances": [
                {"inputs": "What are the common causes of chest pain?"}
            ]
        },
        {
            "instances": [
                "What are the common causes of chest pain?"
            ]
        },
        {
            "prompt": "What are the common causes of chest pain?",
            "parameters": {
                "temperature": 0.7,
                "max_output_tokens": 1000
            }
        },
        {
            "contents": [
                {
                    "parts": [
                        {"text": "What are the common causes of chest pain?"}
                    ]
                }
            ]
        }
    ]
    
    # Test MedGemma endpoint
    medgemma_endpoint = "3612629071499886592.us-central1-369007258962.prediction.vertexai.goog"
    print("=== Testing MedGemma Endpoint ===")
    
    for i, payload in enumerate(payloads_to_try):
        print(f"\n--- Payload {i+1} ---")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        result = test_dedicated_endpoint(medgemma_endpoint, payload)
        if result:
            print(f"SUCCESS: {result}")
            break
    
    # Test Gemma-3 endpoint  
    gemma3_endpoint = "2640414501941280768.us-central1-369007258962.prediction.vertexai.goog"
    print("\n\n=== Testing Gemma-3 Endpoint ===")
    
    for i, payload in enumerate(payloads_to_try):
        print(f"\n--- Payload {i+1} ---")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        result = test_dedicated_endpoint(gemma3_endpoint, payload)
        if result:
            print(f"SUCCESS: {result}")
            break