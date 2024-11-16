# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000 for the Flask app
EXPOSE 8001

# Set environment variables
ENV FLASK_APP=central_api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8001

# Run the application
# idk if the host and port are needed here and if the log level will work lol
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8001", "central_api:app", "--log-level=info"]