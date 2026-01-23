FROM python:3.9-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models

# Generate data, process, and train model during build
RUN python generate_sample_data.py && \
    python src/data_processing.py && \
    python src/train_model.py

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]