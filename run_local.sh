#!/bin/bash

# Nome da imagem
IMAGE_NAME=meu_benchmark

# Diretórios esperados
DATASET_DIR=$(pwd)/app/datasets
OUTPUT_DIR=$(pwd)/app/datasets_and_models_output

# Build da imagem Docker
echo "==> Buildando a imagem Docker: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# Rodando o container com os volumes mapeados
echo "==> Executando o container com os volumes montados"
docker run --rm \
  -v "$DATASET_DIR":/app/datasets \
  -v "$OUTPUT_DIR":/app/datasets_and_models_output \
  $IMAGE_NAME