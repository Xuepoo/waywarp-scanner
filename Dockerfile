FROM python:3.13-slim AS runtime

WORKDIR /app

# Install build dependencies for compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir .

# Remove build dependencies
RUN apt-get purge -y --auto-remove gcc

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["waywarp-scanner"]
