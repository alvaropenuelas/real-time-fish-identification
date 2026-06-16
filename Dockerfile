# Single-stage image for the real-time fish-ID pipeline.
# Base: official Python slim. For GPU deployment, swap to an nvidia/cuda runtime base
# and install CUDA torch wheels; the application code is unchanged.
FROM python:3.12-slim

WORKDIR /app

# System libs: headless OpenCV needs libglib; ffmpeg decodes video sources.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Headless OpenCV belongs in the container (no display). Swap the full GUI build used for
# local dev for opencv-python-headless before installing.
COPY requirements.txt .
RUN sed -i 's/^opencv-python==.*/opencv-python-headless==4.9.0.80/' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to profiling (no GUI). Mount weights/ and a video, e.g.:
#   docker run --rm -v $PWD/weights:/app/weights -v $PWD/clip.mp4:/app/clip.mp4 \
#       fish-id --profile --source clip.mp4 --max-frames 200
ENTRYPOINT ["python", "src/realtime.py"]
CMD ["--help"]
