############################
# 1️⃣  Build stage
############################
FROM python:3.10 AS base

# Avoid Python writing .pyc files + use unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS packages needed by Pillow / Torch vision models
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Set workdir early so later COPY lines are relative
WORKDIR /app

############################
# 2️⃣  Install Python deps
############################
# Copy only requirements first – this allows Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

############################
# 3️⃣  Copy source  🗂️
############################
COPY . .

############################
# 4️⃣  Runtime config
############################
# Port Railway will inject – default to 8000 locally
ENV PORT 8000

EXPOSE ${PORT}

# If you keep your big JSON zipped, unzip here once at build-time
RUN unzip -q data/medical_products_full.zip -d data/ && rm data/medical_products_full.zip

# Gunicorn is optional; Uvicorn is fine for smaller workloads
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
