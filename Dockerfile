# 1. Use an official, slim Python version to keep the file size small
FROM python:3.9-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Install system-level dependencies required for OpenCV
# (OpenCV needs these 'libgl' libraries to handle images)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file into the container
COPY requirements.txt .

# 5. Install the Python libraries (torch, opencv, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your code (src/, data/, etc.) into the container
COPY . .

# 7. (Optional) Set the default command to run your inference script
# This means when someone runs the container, it immediately tries to process data.
CMD ["python", "-m", "src.inference"]