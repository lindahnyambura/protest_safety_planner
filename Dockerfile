# Start with a Python + Conda base image
# FROM continuumio/miniconda3:latest

# # Set the working directory in the container
# WORKDIR /app

# # Copy the environment.yml file to the working directory
# COPY environment.yml .

# # Create Conda environment from the environment.yml file
# RUN conda env create -f environment.yml

# # Make RUN commands use the new environment
# SHELL ["conda", "run", "-n", "protest_safety", "/bin/bash", "-c"]

# # Copy the rest of the application code to the working directory
# COPY . .

# # Activate the environment and run smoke tests when the container starts (default command)
# CMD ["conda", "run", "-n", "protest_safety", "python", "-m", "pytest", "tests/ -v"]

# # Run the MVP script
# #CMD ["conda", "run", "-n", "protest_safety", "python", "src/run_mvp.py"]