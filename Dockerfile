# Use a stable Python base
FROM python:3.12-slim

# Standard Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for many Python libraries like PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (for faster builds)
COPY requirements.txt .
RUN sed -i '/-e ./d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure storage directory exists (Azure containers often need explicit paths)
RUN mkdir -p /app/chroma_data && chmod 777 /app/chroma_data

# Azure defaults to Port 8000 for Python containers
EXPOSE 8000

# Final Command: Run the app and ensure it listens on all interfaces (0.0.0.0)
CMD ["python", "app.py"]