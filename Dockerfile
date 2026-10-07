FROM python:3.11-slim

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Make modules under src/ importable
ENV PYTHONPATH=/app/src

# Working directory inside the container
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the repository into the container
COPY . .

# run_inference.py is located under src/
ENTRYPOINT ["python", "src/run_inference.py"]

# Default to help when no arguments are provided
CMD ["--help"]
