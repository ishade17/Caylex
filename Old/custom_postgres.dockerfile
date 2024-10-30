FROM supabase/postgres:15.6.1.121_arm64

# Install dependencies
RUN apt-get update && apt-get install -y \
    make \
    gcc \
    postgresql-server-dev-15

# Install pgvector
RUN git clone https://github.com/ankane/pgvector.git /pgvector \
    && cd /pgvector \
    && make \
    && make install