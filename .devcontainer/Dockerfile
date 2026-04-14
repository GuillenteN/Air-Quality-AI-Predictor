FROM python:3.11-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# Instalamos dependencias básicas necesarias para las librerías de IA además de git y curl. También borramos archivos temporales para mantener la imagen ligera.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Actualizamos pip.
RUN pip install --upgrade pip

# Copiamos los requirements en un directorio temporal dentro del contenedor para luego instalar las dependencias listadas en ese archivo.
COPY requirements.txt /tmp/requirements.txt

# Instalamos las librerías de IA más populares y útiles para el desarrollo de proyectos de inteligencia artificial.
RUN pip install -r /tmp/requirements.txt

# Creamos el directorio de trabajo dentro del contenedor y lo establecemos como el directorio actual.
WORKDIR /workspace