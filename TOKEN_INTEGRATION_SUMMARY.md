# Token Counter Integration Summary

## Overview
Successfully integrated comprehensive token counting using the tiktoken library throughout the Big5-Agents system to track LLM API usage, costs, and performance.

## Key Components Added

### 1. Token Counter Utility (`utils/token_counter.py`)
- **Purpose**: Central token counting and tracking system
- **Features**:
  - Accurate token counting using tiktoken for different GPT models
  - Thread-safe operation for parallel processing
  - Token limit validation and warnings
  - Comprehensive usage tracking per API call
  - Session-level and question-level aggregation
  - JSON-based storage and logging

### 2. Configuration Updates (`config.py`)
- **Added Token Limits**:
  - `MAX_INPUT_TOKENS = 100000` - Maximum input tokens per API call
  - `MAX_OUTPUT_TOKENS = 8192` - Maximum output tokens per API call
  - `TOKEN_BUDGET_PER_QUESTION = 50000` - Soft limit per question

### 3. Agent Integration (`components/agent.py`)
- **Token Tracking in API Calls**:
  - `_chat_with_timeout()` method now counts input/output tokens
  - `chat()` method includes comprehensive token tracking
  - Token validation before API calls
  - Context tracking for question and simulation IDs
  - Detailed logging of token usage per agent

### 4. MedRAG Integration (`medrag.py`)
- **Enhanced MedRAG Token Tracking**:
  - Token counting in `generate()` method
  - Proper tracking of medical knowledge retrieval costs
  - Integration with global token counter

### 5. Dataset Runner Integration (`dataset_runner.py`)
- **Run-Level Token Management**:
  - Token counter initialization at start of runs
  - Per-question token tracking and storage
  - Comprehensive run summaries with token usage
  - Individual question token files
  - Run-level token summary files

### 6. Simulator Integration (`simulator.py`)
- **Agent Context Setting**:
  - Automatic context setting for all agents
  - Question and simulation ID propagation
  - Enhanced tracking across agent interactions

### 7. Prompt Updates (`utils/prompts.py`)
- **Precision Instructions Added**:
  - "Be precise, concise, and to the point"
  - "Focus directly on the medical/clinical content"
  - "Avoid unnecessary salutations, emotional language, or rejection concerns"
  - "Provide clear, evidence-based reasoning"
  - "Use efficient, professional medical communication"

## File Structure Created

### Token Logs Directory Structure
```
results/
└── [dataset_run_name]/
    ├── token_logs/
    │   └── [detailed_session_logs].json
    ├── run_token_summary.json
    ├── question_[N]_token_usage.json (for each question)
    └── question_[N]_result.json (now includes token_usage field)
```

## Usage and Storage

### Per-Question Tracking
Each question now includes a `token_usage` field with:
- `input_tokens`: Total input tokens for the question
- `output_tokens`: Total output tokens for the question
- `total_tokens`: Sum of input and output tokens
- `api_calls`: Number of API calls made

### Run-Level Summaries
Each run produces:
1. **Individual question token files** - Detailed per-question breakdown
2. **Run token summary** - Aggregated statistics for the entire run
3. **Combined results** - Now includes token usage summary

### Token Counter Features
- **Thread-safe**: Works with parallel question processing
- **Model-aware**: Uses appropriate encoders for different GPT models
- **Context-aware**: Tracks agent roles, question IDs, simulation IDs
- **Comprehensive**: Tracks all API call types (chat, medrag, analysis)

## Key Benefits

1. **Cost Monitoring**: Track exact API costs and usage patterns
2. **Performance Analysis**: Identify token-heavy operations
3. **Budget Management**: Set and monitor token budgets per question/run
4. **Optimization**: Find opportunities to reduce token usage
5. **Compliance**: Ensure token limits are respected
6. **Transparency**: Full visibility into LLM usage across the system

## Backward Compatibility
- All existing functionality preserved
- No changes to function signatures
- Optional token tracking (graceful degradation if tiktoken unavailable)
- Existing results files gain token usage information without breaking changes

## Integration Points

### Primary API Call Tracking
1. `Agent.chat()` - Direct agent conversations
2. `Agent._chat_with_timeout()` - Timeout-protected API calls  
3. `MedRAG.generate()` - Medical knowledge generation

### Storage and Logging
1. Question-level: Individual JSON files per question
2. Run-level: Aggregated summaries and detailed session logs
3. Combined results: Token usage summaries in main results

### Context Propagation
1. Simulator sets agent context (question ID, simulation ID)
2. Agents track context in all API calls
3. Token counter aggregates by context for detailed reporting

## Usage Example

The system now automatically tracks token usage. For a typical run:

```bash
python dataset_runner.py --dataset medqa --num-questions 10
```

This will produce:
- `question_1_token_usage.json` through `question_10_token_usage.json`
- `run_token_summary.json` with aggregated statistics
- Enhanced `combined_results.json` with token usage summaries
- Detailed session logs in `token_logs/` directory

All token tracking is automatic and requires no additional user intervention.