FROM python:3.11-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    curl unzip git openjdk-17-jdk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia apenas o requirements primeiro para otimizar cache
COPY requirements.txt .

# Instala as libs
RUN pip install --no-cache-dir -r requirements.txt

# Agora copia o restante
COPY ./performance_testing ./performance_testing
COPY requirements.txt .
COPY entrypoint.sh .
COPY log4j.properties /app/log4j.properties
ENV SPARK_CONF_DIR=/app

# Dá permissão de execução ao entrypoint
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]