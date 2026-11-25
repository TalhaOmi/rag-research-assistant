FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PAPER_DATA_DIR=/app/data \
    PAPER_CACHE_DIR=/app/.cache \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Editable install so config.ROOT resolves to /app.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[embed]"

# The corpus and index are built at image build time (needs internet: arXiv + Hugging Face),
# so the running container starts instantly and needs no PDFs in git.
COPY data/papers.csv ./data/papers.csv
RUN python -m paper_assistant fetch && python -m paper_assistant build-index

COPY start.sh ./start.sh
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["sh", "start.sh"]
