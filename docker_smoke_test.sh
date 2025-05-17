#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ramco"

echo "🛠 Building Docker image '${IMAGE_NAME}'…"
docker build -t "${IMAGE_NAME}" .

echo
echo "🐳 Running smoke test inside the container…"
docker run --rm "${IMAGE_NAME}"

echo
echo "✅ All smoke tests passed!"

