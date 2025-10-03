# ---------------------------------
# STAGE 1: BUILDER (DEPENDENCIES AND MODEL DOWNLOAD)
# ---------------------------------
FROM python:3.10-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model Caching Setup
ENV MODEL_NAME facebook/bart-large-cnn
ENV HF_HOME /root/.cache/huggingface

# Download model weights to the cache during BUILD TIME (CRITICAL for speed)
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
               AutoModelForSeq2SeqLM.from_pretrained('$MODEL_NAME'); \
               AutoTokenizer.from_pretrained('$MODEL_NAME');"

COPY ./app ./app
COPY ./model ./model

# ---------------------------------
# STAGE 2: RUNTIME (MINIMAL EXECUTION ENVIRONMENT)
# ---------------------------------
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin 
COPY --from=builder /app /app 

# CRITICAL FIX: Copy the Hugging Face cache containing model weights
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

ENV HF_HOME /root/.cache/huggingface
ENV PORT 8000

EXPOSE 8000

# Start application using Gunicorn (workers=1 to manage BART-large RAM efficiently)
CMD ["gunicorn", "app.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
