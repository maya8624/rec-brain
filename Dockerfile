# App image — built on top of rec-brain-base which has all dependencies pre-installed
# Build base first with: docker build -f Dockerfile.base -t rec-brain-base .
# Then run: docker compose up --build

# Use the pre-built base image with all dependencies already installed
FROM rec-brain-base

# Set the working directory
WORKDIR /app

COPY . .

EXPOSE 8000

# Start the FastAPI app — 0.0.0.0 binds to all network interfaces inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
