#!/bin/sh
# API on an internal port, Streamlit UI on the public port (7860 is what Hugging Face Spaces expects).
set -e
uvicorn paper_assistant.api:app --host 127.0.0.1 --port 8000 &
exec streamlit run src/paper_assistant/ui.py \
  --server.port 7860 --server.address 0.0.0.0 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false
