# API Integration Guide

## {CLIENT_NAME}

Your private AI API is ready.

### Endpoint

```
https://{CLIENT_SUBDOMAIN}.yourdomain.com
```

### Authentication

Include your API key in every request:

```
X-API-Key: {CLIENT_API_KEY}
```

### Generate endpoint

**POST** `/generate`

Request:

```json
{
  "prompt": "your text here",
  "max_tokens": 256,
  "temperature": 0.7
}
```

Response:

```json
{
  "job_id": "abc-123",
  "status": "queued",
  "poll_url": "/result/abc-123"
}
```

### Poll for result

**GET** `/result/{job_id}`

Response when ready:

```json
{
  "status": "done",
  "output": "AI response here",
  "tokens_in": 120,
  "tokens_out": 87,
  "latency_ms": 340
}
```

### Health check

**GET** `/health`

- No authentication required
- Use for your own uptime monitoring (e.g. UptimeRobot)

### Python example

```python
import httpx
import time

BASE = "https://{CLIENT_SUBDOMAIN}.yourdomain.com"
KEY = "{CLIENT_API_KEY}"


def generate(prompt: str) -> str:
    headers = {"X-API-Key": KEY}

    response = httpx.post(
        f"{BASE}/generate",
        json={"prompt": prompt, "max_tokens": 256, "temperature": 0.7},
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]

    while True:
        result = httpx.get(
            f"{BASE}/result/{job_id}",
            headers=headers,
            timeout=30.0,
        ).json()

        if result["status"] == "done":
            return result["output"]
        if result["status"] == "error":
            raise RuntimeError(result.get("error", "Generation failed"))
        time.sleep(1)
```

### Rate limits

Your current plan:

- **{RATE_LIMIT}** requests per minute
- **{MONTHLY_LIMIT}** requests per month
- **{TOKEN_LIMIT}** tokens per month

### Support

- Email: support@yourdomain.com
- Response time: within 24 hours
