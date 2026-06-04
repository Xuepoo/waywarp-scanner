FROM python:3.13-slim AS runtime

WORKDIR /app

# Install build dependencies for compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Pre-install dependencies to utilize Docker cache layers
COPY pyproject.toml README.md .
RUN mkdir waywarp_scanner && touch waywarp_scanner/__init__.py
RUN pip install --no-cache-dir .

# Copy source code and reinstall using --no-deps
COPY . .
RUN pip install --no-cache-dir --no-deps .

# Remove build dependencies
RUN apt-get purge -y --auto-remove gcc

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["waywarp-scanner"]
