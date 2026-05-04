# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables to avoid writing .pyc files and to ensure output is sent straight to terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV/PyTorch/Graphviz if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# We also ensure the necessary DL libraries are present
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision timm shap scipy h5py streamlit graphviz

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
