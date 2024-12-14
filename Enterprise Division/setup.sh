#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Optional: Clone the Supabase repository if it doesn't exist
if [ ! -d "supabase" ]; then
  echo "Cloning Supabase repository..."
  git clone --depth 1 https://github.com/supabase/supabase
fi

# Copy your docker-compose.override.yml into the supabase/docker directory
if [ -f "docker-compose.override.yaml" ]; then
  echo "Copying docker-compose.override.yml..."
  cp docker-compose.override.yaml supabase/docker/
else
  echo "Error: docker-compose.override.yaml not found!"
  exit 1
fi

# Copy your .env file into the supabase/docker directory
if [ -f ".env" ]; then
  echo "Copying .env file..."
  cp .env supabase/docker/
else
  echo "Error: .env file not found!"
  exit 1
fi

# Navigate to the supabase/docker directory
cd supabase/docker

# Pull the latest Docker images
echo "Pulling latest Docker images..."
docker compose pull

# Build and start the Supabase services along with your division-api
echo "Building and starting Docker containers..."
docker compose up --build -d



# #!/bin/bash

# # Ensure the script exits if any command fails
# set -e

# # Clone the Supabase repository
# #git clone --depth 1 https://github.com/supabase/supabase

# # Copy your docker-compose.override.yml into the supabase/docker directory
# cp docker-compose.override.yaml supabase/docker/

# # Copy your .env file into the supabase/docker directory
# cp .env supabase/docker/

# # Navigate to the supabase/docker directory
# cd supabase/docker

# # Pull the latest Docker images
# docker compose pull

# # Build and start the Supabase services along with your division-api
# docker compose up --build
