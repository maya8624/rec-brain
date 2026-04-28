# App image — built on top of rec-brain-base which has all dependencies pre-installed
# Build base first with: docker build -f Dockerfile.base -t rec-brain-base .
# Then run: docker compose up --build

# Use the pre-built base image with all dependencies already installed
ARG BASE_IMAGE=rec-brain-base
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /app

# Install lighter dependencies — rebuilt whenever requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY . .

EXPOSE 8000

# Start the FastAPI app — 0.0.0.0 binds to all network interfaces inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
