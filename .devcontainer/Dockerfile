FROM python:3.11-bullseye

# Instala dependencias básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Actualiza pip e instala librerías de IA
RUN pip install --upgrade pip && \
    pip install numpy pandas matplotlib scikit-learn tensorflow keras \
    seaborn shap xgboost lightgbm bokeh scipy NLTK ipykernel panel \
    lxml openpyxl torch transformers sentencepiece 
