# LLM Summarization API (FastAPI + BART-Large)
This repository provides a functional microservice for **Abstractive Text Summarization**. It is developed for consistent performance and uses a **BART-Large** LLM integrated with **FastAPI**, **Docker**, and **Gunicorn** for deployment.

### Setup and Running
The project is designed to be run using Docker-Compose.

#### 1. Prerequisites
- Docker
- Docker Compose
- Sufficient RAM (A minimum of **12 GB RAM** is recommended for the Docker Engine/VM to safely load the model.)

#### 2. Clone the Repository
```bash
git clone abstractive-summarizer-api
cd abstractive-summarizer-api
```

#### 3. Build and Start the Service
The following command builds the Docker image, caches the model weights inside the image, and starts the API service.

**Note**: The initial build process may take some time due to the large size of the model. Grab a ☕!
```bash
make run
```
#### 4. Check Status
Ensure the container is running:

```bash
make logs
```

### API Usage
The API is available on port 8000 of your local machine.

#### A. Liveness Check (Service Running)
Checks if the service is alive. ✅

- **Endpoint**: `GET /health`
- **Successful Response** (reflects the StatusOutput schema):
```json
{
    "status": "ok",
    "model_status": "Loading (Awaiting first request)"
}
```
#### B. Readiness Check (Model Status)
Checks the precise loading status of the AI model. This is the official readiness endpoint.
- **Endpoint**: `GET /api/v1/status`
- **Successful Response**: Returns `{"model_status": "Loaded"}` once the model is in memory.

#### C. Text Summarization (Inference)
Used to summarize a long text.

- **Endpoint**: `POST /api/v1/summarize`
- **Content-Type**: `application/json`
- **Request Body** (Note: Pydantic enforces minimum text length):
```json
{
  "text": "The long text goes here...",
  "min_length": 30,
  "max_length": 150
}
```
- **Successful Response**:
```json
{
  "summary": "The generated, concise, and coherent summary text."
}
```
#### D. Interactive Documentation (Swagger UI)
You can view all API schemas and the interactive testing interface in your browser: 🔎
```
http://localhost:8000/docs
```
### Quality Assurance & Testing
#### 1. Code Formatting
Checks and automatically fixes code styling (using Black) and import ordering (using Isort).
```bash
make format
```
#### 2. Run Tests
Executes unit tests using `pytest` to verify core functionality, DI, and error handling. The AI model is mocked during these tests for speed and reliability.
```bash
make test
```

### Stopping the Service
Stops and removes all containers and the network created by Docker Compose:
```bash
make clean
```
