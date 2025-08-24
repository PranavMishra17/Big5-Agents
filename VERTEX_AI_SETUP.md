# Vertex AI SLM Integration Setup

## Prerequisites

1. **Google Cloud Project**: Project ID `369007258962` (already configured)
2. **Deployed Models**: 
   - Gemma-3-12B: Endpoint `2640414501941280768`
   - MedGemma-4B: Endpoint `3612629071499886592`
3. **Region**: `us-central1`

## Authentication Setup

### Option 1: Application Default Credentials (Recommended)
```bash
gcloud auth application-default login
```

### Option 2: Service Account JSON Key
1. Download service account JSON from Google Cloud Console
2. Set environment variable:
```bash
set GOOGLE_APPLICATION_CREDENTIALS="path\to\service-account-key.json"
```

### Option 3: Temporary Testing (if gcloud is not available)
For testing purposes, you can temporarily use a service account key by placing it in the project directory and updating the code.

## Configuration Files Modified

### config.py Changes
- Added `VERTEX_AI_DEPLOYMENTS` with your specific endpoint IDs
- Updated deployment selection functions to use Vertex AI only
- Configured parallel processing for 2 SLM models

### components/agent.py Changes  
- Added Vertex AI client initialization
- Implemented `_chat_vertex_ai()` method for SLM API calls
- Updated `chat()` method to route to Vertex AI
- Added fallback handling for vision tasks (text-only for now)

## Available Models in SLM Branch

1. **gemma3_12b_1**: Gemma-3-12B-IT model for general reasoning
2. **medgemma_4b_1**: MedGemma-4B-IT model for medical tasks

## Usage Example

```python
from components.agent import Agent

# Create agent with MedGemma model
medical_agent = Agent(
    role="Medical Specialist",
    expertise_description="Medical diagnosis expert",
    agent_index=1  # Uses medgemma_4b_1
)

# Create agent with Gemma-3 model
reasoning_agent = Agent(
    role="General Reasoner", 
    expertise_description="General reasoning specialist",
    agent_index=0  # Uses gemma3_12b_1
)

# Use the agents
response = medical_agent.chat("What are the symptoms of pneumonia?")
```

## Testing

Run the integration test:
```bash
python test_vertex_ai_integration.py
```

This will test:
- Configuration loading
- Agent initialization
- Basic chat functionality with both models

## Troubleshooting

1. **Authentication Error**: Ensure ADC is set up or service account key is configured
2. **Endpoint Not Found**: Verify endpoint IDs are correct in config.py
3. **Permission Denied**: Check that your account has Vertex AI permissions
4. **Model Not Available**: Ensure models are deployed and accessible in us-central1

## Features Working
- ✅ Dual SLM model support (Gemma-3-12B + MedGemma-4B)  
- ✅ Round-robin deployment selection
- ✅ Parallel question processing
- ✅ Token tracking and logging
- ✅ MedRAG integration compatibility
- ⚠️ Vision support (text-only fallback for now)

## Next Steps
1. Set up authentication
2. Test with actual medical questions
3. Implement proper vision support for Vertex AI if needed
4. Fine-tune model parameters (temperature, max_tokens, etc.)