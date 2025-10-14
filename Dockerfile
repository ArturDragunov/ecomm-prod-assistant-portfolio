FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install git and uv
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install uv

# Copy source code (first dot) and paste to the current directory (/app)
COPY . .

# Now build local package
RUN uv sync --frozen          

EXPOSE 8000

# Use uv run - no MCP server needed
CMD ["uv", "run", "uvicorn", "prod_assistant.router.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# run uvicorn properly on 0.0.0.0:8000
# Docker with MCP server -> run server first and then uvicorn
# CMD ["bash", "-c", "python prod_assistant/mcp_servers/product_search_server.py & uvicorn prod_assistant.router.main:app --host 0.0.0.0 --port 8000 --workers 2"]
