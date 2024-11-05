#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Clone the Supabase repository
#git clone --depth 1 https://github.com/supabase/supabase

# Copy your docker-compose.override.yml into the supabase/docker directory
cp docker-compose.override.yaml supabase/docker/

# Copy your .env file into the supabase/docker directory
cp .env supabase/docker/

# Navigate to the supabase/docker directory
cd supabase/docker

# Pull the latest Docker images
docker compose pull

# Build and start the Supabase services along with your division-api
docker compose up --build
