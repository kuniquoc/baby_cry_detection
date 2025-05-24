FROM ubuntu:22.04 as builder

# Set build-time variables
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.13.0
ARG TZ=UTC

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION}*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir wheel \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Create necessary directories and set permissions
RUN mkdir -p /app/static /app/templates /app/data && \
    chown -R nobody:nobody /app

# Copy source code for potential build dependencies
COPY --chown=nobody:nobody core/ ./core/
COPY --chown=nobody:nobody models/ ./models/
COPY --chown=nobody:nobody services/ ./services/
COPY --chown=nobody:nobody utils/ ./utils/
COPY --chown=nobody:nobody websocket/ ./websocket/
COPY --chown=nobody:nobody static/ ./static/
COPY --chown=nobody:nobody templates/ ./templates/
COPY --chown=nobody:nobody app.py .
COPY --chown=nobody:nobody fcm_service.py .
COPY --chown=nobody:nobody firebase_service.py .

# Final stage
FROM gcr.io/distroless/base-debian11:nonroot

# Set environment variables
ENV PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Copy Python from builder
COPY --from=builder /usr/local/bin/python* /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/lib/libpython3.13.so* /usr/local/lib/
COPY --from=builder /usr/local/include/python3.13 /usr/local/include/python3.13

# Copy virtual environment and necessary files from builder
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories in final stage
USER root
RUN mkdir -p /app/static /app/templates /app/data && \
    chown -R nonroot:nonroot /app
USER nonroot

# Copy application code and static files
COPY --from=builder --chown=nonroot:nonroot /app/core ./core
COPY --from=builder --chown=nonroot:nonroot /app/models ./models
COPY --from=builder --chown=nonroot:nonroot /app/services ./services
COPY --from=builder --chown=nonroot:nonroot /app/utils ./utils
COPY --from=builder --chown=nonroot:nonroot /app/websocket ./websocket
COPY --from=builder --chown=nonroot:nonroot /app/static ./static
COPY --from=builder --chown=nonroot:nonroot /app/templates ./templates
COPY --from=builder --chown=nonroot:nonroot /app/static ./static
COPY --from=builder --chown=nonroot:nonroot /app/templates ./templates
COPY --from=builder --chown=nonroot:nonroot /app/app.py .
COPY --from=builder --chown=nonroot:nonroot /app/fcm_service.py .
COPY --from=builder --chown=nonroot:nonroot /app/firebase_service.py .

# Create necessary directories in builder stage
RUN mkdir -p /app/static /app/templates

# Copy system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsndfile.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libFLAC.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libvorbis.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libogg.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libvorbisenc.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/bin/ffmpeg /usr/bin/

# Update dynamic linker
COPY --from=builder /etc/ld.so.cache /etc/ld.so.cache
COPY --from=builder /lib/x86_64-linux-gnu/ld-linux-x86-64.so* /lib/x86_64-linux-gnu/

# Copy application files
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Create necessary directories with correct permissions
RUN mkdir -p static templates models

# Copy application code
COPY --from=builder /app/core ./core/
COPY --from=builder /app/models ./models/
COPY --from=builder /app/services ./services/
COPY --from=builder /app/utils ./utils/
COPY --from=builder /app/websocket ./websocket/
COPY --from=builder /app/app.py .
COPY --from=builder /app/fcm_service.py .
COPY --from=builder /app/firebase_service.py .

# Copy static files and templates
COPY static/ ./static/
COPY templates/ ./templates/

# Expose the port the app runs on
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
