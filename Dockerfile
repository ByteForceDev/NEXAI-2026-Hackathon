# Dockerfile for NEXAI 2026 Hackathon - Autonomous Dynamic Route Engine
FROM python:3.11-slim

# Set environment variables for performance and unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_THEME_BASE=light

WORKDIR /app

# Install build dependencies for C++ pyvroom compilation if wheel build is required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and assets
COPY . .

# Expose default Hugging Face Spaces / Streamlit port
EXPOSE 7860

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
