# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn
# RUN pip install gunicorn

# Copy the rest of the application code
COPY . .

# Expose port 8000 for the Flask app
EXPOSE 8000

# Run the application using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "agent_comm_api:app"]







# # Use an official Python runtime as a parent image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container
# COPY requirements.txt ./

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Expose port 8000 for the Flask app
# EXPOSE 8000

# # Set environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=8000

# # Run the application
# CMD ["flask", "run"]
