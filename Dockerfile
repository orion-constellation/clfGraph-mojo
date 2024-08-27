FROM python:3.11-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.3.2 \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libssl-dev \
    libffi-dev \
    libgmp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Mojo
RUN curl -sSL https://get.mojo-lang.org | sh

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-root --only main

# Copy the application code
COPY . .

# Build Mojo custom operations
RUN mojo build --release

# Expose application port
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]