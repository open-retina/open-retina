# Container Usage Instructions

## Prerequisites

### For Docker

- Docker installed on your system
- NVIDIA Container Toolkit (for GPU support)

### For Singularity

- Singularity/Apptainer installed on your system
- NVIDIA drivers installed on the host system

## Building the Containers

### Docker

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your-org/openretina.git
cd openretina

# Build the Docker image
docker build -t openretina .
```

### Singularity

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your-org/openretina.git
cd openretina

# Build the Singularity image
singularity build --fakeroot openretina.sif Singularity.def
```

## Running Tests

### Docker

```bash
# Run tests using make test-all
docker run --gpus all -v $(pwd):/openretina openretina make test-all

# If you need to run specific test files or with specific options
docker run --gpus all -v $(pwd):/openretina openretina pytest tests/specific_test.py
```

### Singularity

```bash
# Run tests using make test-all
singularity exec --nv openretina.sif make test-all

# If you need to run specific test files or with specific options
singularity exec --nv openretina.sif pytest tests/specific_test.py
```

## Running Jupyter Lab

### Docker

```bash
# Run Jupyter Lab on port 8888
docker run --gpus all \
    -v $(pwd):/openretina \
    -p 8888:8888 \
    openretina \
    jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

# For a specific port (e.g., 9999)
docker run --gpus all \
    -v $(pwd):/openretina \
    -p 9999:8888 \
    openretina \
    jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

### Singularity

```bash
# Run Jupyter Lab on port 8888
singularity exec --nv \
    openretina.sif \
    jupyter lab --ip 0.0.0.0 --port 8888 --no-browser

# For a specific port (e.g., 9999)
singularity exec --nv \
    openretina.sif \
    jupyter lab --ip 0.0.0.0 --port 9999 --no-browser
```

## Additional Notes

1. Volume Mounting
   - The `-v $(pwd):/openretina` (Docker) and automatic home directory mounting (Singularity) ensure that your local changes are reflected inside the container
   - Any files created in the container will be available in your local directory

2. GPU Access
   - `--gpus all` (Docker) and `--nv` (Singularity) flags enable GPU access
   - Make sure your NVIDIA drivers are properly installed on the host system

3. Jupyter Lab Access
   - After starting Jupyter Lab, look for a URL in the terminal output
   - Copy the URL with the token and paste it into your browser
   - The URL will look like: `http://127.0.0.1:8888/lab?token=<your_token>`

4. Common Issues
   - If the port is already in use, try a different port number
   - For Docker, if you get permission errors, make sure your user is in the docker group
   - For Singularity, if GPU access fails, verify your NVIDIA drivers are properly installed

## Environment Variables

You can pass additional environment variables to the containers:

### Docker

```bash
docker run --gpus all -v $(pwd):/openretina -e CUDA_VISIBLE_DEVICES=0 openretina make test-all
```

### Singularity

```bash
CUDA_VISIBLE_DEVICES=0 singularity exec --nv openretina.sif make test-all
```
