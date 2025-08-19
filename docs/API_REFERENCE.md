# iLLuMinator 4.9B API Reference

Complete API documentation for all iLLuMinator 4.9B interfaces, including REST API, Python SDK, and CLI tools.

## Table of Contents

- [REST API](#rest-api)
- [Python SDK](#python-sdk)
- [CLI Tools](#cli-tools)
- [WebSocket API](#websocket-api)
- [Batch Processing API](#batch-processing-api)
- [Model Management API](#model-management-api)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## REST API

### Base URL

```
Production: https://api.illuminator.ai
Development: http://localhost:8000
```

### Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.illuminator.ai/generate
```

### Endpoints

#### Text Generation

**POST** `/generate`

Generate text continuation from a prompt.

##### Request Body

```json
{
  "prompt": "string",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.0,
  "stop_sequences": [".", "!", "?"],
  "return_full_text": false,
  "stream": false
}
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | Required | Input text prompt |
| `max_tokens` | integer | 100 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling parameter |
| `top_k` | integer | 50 | Top-k sampling parameter |
| `repetition_penalty` | float | 1.0 | Repetition penalty (1.0-2.0) |
| `stop_sequences` | array | [] | Sequences to stop generation |
| `return_full_text` | boolean | false | Return prompt + generated text |
| `stream` | boolean | false | Stream response tokens |

##### Response

```json
{
  "generated_text": "The generated continuation...",
  "tokens_generated": 45,
  "generation_time": 1.23,
  "model_version": "4.9B",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 45,
    "total_tokens": 55
  }
}
```

##### Example

```bash
curl -X POST "https://api.illuminator.ai/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

#### Chat Completion

**POST** `/chat/completions`

OpenAI-compatible chat completion endpoint.

##### Request Body

```json
{
  "model": "illuminator-4.9b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

##### Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "illuminator-4.9b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 12,
    "total_tokens": 32
  }
}
```

#### Health Check

**GET** `/health`

Check API server health status.

##### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "memory_usage": {
    "gpu_memory_used": "12.5GB",
    "gpu_memory_total": "24GB"
  },
  "uptime": "2h 30m 15s",
  "version": "1.0.0"
}
```

#### Model Information

**GET** `/info`

Get detailed model information.

##### Response

```json
{
  "model_name": "iLLuMinator 4.9B",
  "version": "1.0.0",
  "parameters": "4.9B",
  "architecture": {
    "type": "Transformer",
    "layers": 32,
    "heads": 32,
    "d_model": 4096,
    "vocabulary_size": 65536,
    "context_length": 4096
  },
  "capabilities": [
    "text_generation",
    "chat_completion",
    "code_generation",
    "instruction_following"
  ],
  "training_data": {
    "sources": ["OpenOrca", "UltraChat", "CodeAlpaca"],
    "total_tokens": "2.5T",
    "languages": ["English", "Code"]
  }
}
```

#### Tokenization

**POST** `/tokenize`

Tokenize text using the model's tokenizer.

##### Request Body

```json
{
  "text": "Hello, world!"
}
```

##### Response

```json
{
  "tokens": [15496, 11, 1917, 0],
  "token_count": 4,
  "text_length": 13
}
```

#### Detokenization

**POST** `/detokenize`

Convert tokens back to text.

##### Request Body

```json
{
  "tokens": [15496, 11, 1917, 0]
}
```

##### Response

```json
{
  "text": "Hello, world!",
  "token_count": 4
}
```

### Streaming Responses

Enable streaming for real-time token generation:

```bash
curl -X POST "https://api.illuminator.ai/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "stream": true}' \
  --no-buffer
```

Stream format (Server-Sent Events):

```
data: {"token": "Once", "index": 0}

data: {"token": " upon", "index": 1}

data: {"token": " a", "index": 2}

data: [DONE]
```

## Python SDK

### Installation

```bash
pip install illuminator-sdk
```

### Basic Usage

```python
from illuminator import IlluminatorClient

# Initialize client
client = IlluminatorClient(
    api_key="your_api_key",
    base_url="https://api.illuminator.ai"
)

# Generate text
response = client.generate(
    prompt="Explain machine learning:",
    max_tokens=100,
    temperature=0.7
)

print(response.generated_text)
```

### Advanced Usage

```python
import asyncio
from illuminator import AsyncIlluminatorClient

async def main():
    client = AsyncIlluminatorClient(api_key="your_api_key")
    
    # Async generation
    response = await client.generate(
        prompt="Write a Python function:",
        max_tokens=200,
        temperature=0.3
    )
    
    # Streaming generation
    async for token in client.generate_stream(
        prompt="Tell me a story:",
        max_tokens=500
    ):
        print(token.text, end="", flush=True)
    
    await client.close()

asyncio.run(main())
```

### Chat Interface

```python
from illuminator import ChatClient

chat = ChatClient(api_key="your_api_key")

# Start conversation
chat.add_message("user", "Hello! How are you?")
response = chat.complete()
print(response.message.content)

# Continue conversation
chat.add_message("user", "Can you help me with Python?")
response = chat.complete(temperature=0.5)
print(response.message.content)

# Get conversation history
history = chat.get_history()
```

### Batch Processing

```python
from illuminator import BatchClient

batch = BatchClient(api_key="your_api_key")

# Submit batch job
prompts = [
    "Summarize: The quick brown fox...",
    "Translate to French: Hello world",
    "Code review: def fibonacci(n):"
]

job = batch.submit_job(
    prompts=prompts,
    max_tokens=100,
    temperature=0.7
)

# Monitor progress
while not job.is_complete():
    print(f"Progress: {job.progress}%")
    time.sleep(5)

# Get results
results = job.get_results()
for i, result in enumerate(results):
    print(f"Prompt {i}: {result.generated_text}")
```

### Error Handling

```python
from illuminator import IlluminatorClient, IlluminatorError
from illuminator.exceptions import (
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError
)

client = IlluminatorClient(api_key="your_api_key")

try:
    response = client.generate("Hello world")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
except ModelNotFoundError:
    print("Model not available")
except IlluminatorError as e:
    print(f"API error: {e}")
```

### Configuration

```python
from illuminator import IlluminatorClient, Config

# Custom configuration
config = Config(
    api_key="your_api_key",
    base_url="https://api.illuminator.ai",
    timeout=60,
    max_retries=3,
    retry_delay=1.0
)

client = IlluminatorClient(config=config)
```

## CLI Tools

### Installation

The CLI tools are included with the main package:

```bash
pip install illuminator-4.9b
```

### Text Generation

```bash
# Basic generation
illuminator generate "Explain quantum physics:" --max-tokens 200

# With custom parameters
illuminator generate "Write code:" \
    --max-tokens 300 \
    --temperature 0.3 \
    --top-p 0.9 \
    --output output.txt

# Interactive mode
illuminator interactive --model illuminator-4.9b
```

### Chat Mode

```bash
# Start chat session
illuminator chat

# Chat with custom settings
illuminator chat \
    --temperature 0.7 \
    --max-tokens 150 \
    --system-message "You are a helpful coding assistant"
```

### Batch Processing

```bash
# Process file of prompts
illuminator batch prompts.txt --output results.jsonl

# Parallel processing
illuminator batch prompts.txt \
    --workers 4 \
    --batch-size 10 \
    --output results.jsonl
```

### Model Management

```bash
# List available models
illuminator models list

# Download model
illuminator models download illuminator-4.9b

# Model information
illuminator models info illuminator-4.9b

# Start local server
illuminator serve \
    --model checkpoints/best_model.pt \
    --host 0.0.0.0 \
    --port 8000
```

### Configuration

```bash
# Set API key
illuminator config set api_key YOUR_API_KEY

# Set default model
illuminator config set model illuminator-4.9b

# View configuration
illuminator config list

# Reset configuration
illuminator config reset
```

## WebSocket API

### Connection

```javascript
const socket = new WebSocket('wss://api.illuminator.ai/ws');

// Authentication
socket.onopen = () => {
    socket.send(JSON.stringify({
        type: 'auth',
        token: 'YOUR_API_KEY'
    }));
};
```

### Real-time Generation

```javascript
// Start generation
socket.send(JSON.stringify({
    type: 'generate',
    data: {
        prompt: "Once upon a time",
        max_tokens: 100,
        temperature: 0.7
    }
}));

// Receive tokens
socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'token':
            console.log(message.data.token);
            break;
        case 'complete':
            console.log('Generation complete');
            break;
        case 'error':
            console.error(message.data.error);
            break;
    }
};
```

### Chat Interface

```javascript
// Send chat message
socket.send(JSON.stringify({
    type: 'chat',
    data: {
        messages: [
            {role: 'user', content: 'Hello!'}
        ],
        temperature: 0.7
    }
}));
```

## Batch Processing API

### Submit Batch Job

**POST** `/batch/jobs`

```json
{
  "name": "my_batch_job",
  "prompts": [
    {"prompt": "Summarize this text...", "max_tokens": 100},
    {"prompt": "Translate to Spanish...", "max_tokens": 50}
  ],
  "default_params": {
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

### Get Job Status

**GET** `/batch/jobs/{job_id}`

```json
{
  "id": "job_123",
  "name": "my_batch_job",
  "status": "running",
  "progress": 65,
  "created_at": "2024-01-15T10:00:00Z",
  "started_at": "2024-01-15T10:01:00Z",
  "total_prompts": 100,
  "completed_prompts": 65,
  "failed_prompts": 2,
  "estimated_completion": "2024-01-15T10:15:00Z"
}
```

### Get Job Results

**GET** `/batch/jobs/{job_id}/results`

```json
{
  "job_id": "job_123",
  "results": [
    {
      "index": 0,
      "status": "completed",
      "generated_text": "Summary of the text...",
      "tokens_generated": 85,
      "generation_time": 1.2
    },
    {
      "index": 1,
      "status": "failed",
      "error": "Input too long"
    }
  ]
}
```

## Model Management API

### List Models

**GET** `/models`

```json
{
  "models": [
    {
      "id": "illuminator-4.9b",
      "name": "iLLuMinator 4.9B",
      "description": "Full 4.9B parameter model",
      "parameters": "4.9B",
      "context_length": 4096,
      "status": "available"
    },
    {
      "id": "illuminator-practical",
      "name": "iLLuMinator Practical",
      "description": "Lightweight 120M parameter model",
      "parameters": "120M",
      "context_length": 1024,
      "status": "available"
    }
  ]
}
```

### Model Statistics

**GET** `/models/{model_id}/stats`

```json
{
  "model_id": "illuminator-4.9b",
  "usage_stats": {
    "total_requests": 1500,
    "total_tokens_generated": 450000,
    "average_generation_time": 1.8,
    "error_rate": 0.02
  },
  "performance_metrics": {
    "throughput": "25 tokens/second",
    "memory_usage": "18.5GB",
    "gpu_utilization": 85
  }
}
```

## Authentication

### API Key Authentication

Include your API key in the Authorization header:

```bash
curl -H "Authorization: Bearer sk-1234567890abcdef..." \
     https://api.illuminator.ai/generate
```

### JWT Token Authentication

For enterprise users, JWT tokens are supported:

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
     https://api.illuminator.ai/generate
```

### OAuth 2.0 (Enterprise)

For enterprise integrations, OAuth 2.0 is available:

```bash
# Get access token
curl -X POST https://auth.illuminator.ai/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET"

# Use access token
curl -H "Authorization: Bearer ACCESS_TOKEN" \
     https://api.illuminator.ai/generate
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Model or endpoint not found |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model overloaded |

### Error Response Format

```json
{
  "error": {
    "type": "validation_error",
    "code": "invalid_parameter",
    "message": "The parameter 'temperature' must be between 0.0 and 2.0",
    "param": "temperature",
    "request_id": "req_1234567890"
  }
}
```

### Common Error Types

- `authentication_error`: Invalid API key
- `rate_limit_error`: Too many requests
- `validation_error`: Invalid request parameters
- `model_error`: Model processing error
- `timeout_error`: Request timeout
- `quota_exceeded`: Usage quota exceeded

## Rate Limiting

### Default Limits

| Tier | Requests/Minute | Tokens/Hour |
|------|----------------|-------------|
| Free | 60 | 50,000 |
| Basic | 300 | 250,000 |
| Pro | 1,500 | 1,000,000 |
| Enterprise | Custom | Custom |

### Rate Limit Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
X-RateLimit-Reset-Tokens: 1640995800
```

### Handling Rate Limits

```python
import time
from illuminator import IlluminatorClient, RateLimitError

client = IlluminatorClient(api_key="your_api_key")

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.generate(prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                time.sleep(e.retry_after)
            else:
                raise
```

## Examples

### Code Generation

```bash
curl -X POST "https://api.illuminator.ai/generate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci numbers:",
    "max_tokens": 200,
    "temperature": 0.3,
    "stop_sequences": ["\n\n"]
  }'
```

### Creative Writing

```python
from illuminator import IlluminatorClient

client = IlluminatorClient(api_key="your_api_key")

story = client.generate(
    prompt="Write a short story about a robot discovering emotions:",
    max_tokens=500,
    temperature=0.8,
    top_p=0.9
)

print(story.generated_text)
```

### Question Answering

```bash
curl -X POST "https://api.illuminator.ai/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "illuminator-4.9b",
    "messages": [
      {
        "role": "system",
        "content": "You are a knowledgeable assistant. Provide accurate, helpful answers."
      },
      {
        "role": "user",
        "content": "What is the capital of France and what is it famous for?"
      }
    ],
    "max_tokens": 150,
    "temperature": 0.5
  }'
```

### Data Analysis

```python
from illuminator import IlluminatorClient

client = IlluminatorClient(api_key="your_api_key")

analysis = client.generate(
    prompt="""
    Analyze this sales data and provide insights:
    Q1: $125,000
    Q2: $150,000
    Q3: $140,000
    Q4: $180,000
    
    Analysis:
    """,
    max_tokens=300,
    temperature=0.4
)

print(analysis.generated_text)
```

This API reference provides comprehensive documentation for all iLLuMinator 4.9B interfaces. For additional examples and tutorials, visit our documentation website.