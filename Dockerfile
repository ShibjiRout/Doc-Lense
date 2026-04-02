FROM python:3.12-slim

# Prevent Python from writing .pyc files and keep logs flushing to the terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for PyMuPDF and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Create the persistent directory for ChromaDB
RUN mkdir -p /app/chroma_data && chmod 777 /app/chroma_data

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Streamlit uses the port provided by Render ($PORT)
# Note: We don't hardcode EXPOSE 8000 here because Render handles the mapping
EXPOSE 8501

# The Start Command: Run Streamlit and bind to $PORT
CMD ["sh", "-c", "streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0"]