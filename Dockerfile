FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create mount point for the volume
VOLUME ["/app/polygon_cache"]

# Define required environment variables (must be provided at runtime)
# No default values for security best practices
ENV API_KEY="" \
    PG_HOST="" \
    PG_PORT="" \
    PG_DATABASE="" \
    PG_USER="" \
    PG_PASSWORD=""

# Default command to run main.py (single backtest)
CMD ["python", "main.py"] 