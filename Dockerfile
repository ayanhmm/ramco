# Use the same base
FROM python:3.11-slim

# Avoid interactive prompts, set UTF-8 locale
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

WORKDIR /app

# 1. Install OS deps for OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 2. Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy code
COPY . .

# 4. Ensure your smoke script is executable
RUN chmod +x scripts/utils/tests/smoke_test_1_cv.sh

# 5. Default to running your shell-based smoke test
CMD ["bash", "-lc", "scripts/utils/tests/smoke_test_1_cv.sh"]

