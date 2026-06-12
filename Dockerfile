FROM python:3.12-slim

WORKDIR /app

# Instalar dependências da API
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar a aplicação FastAPI
COPY main.py .

# Copiar o modelo treinado
# Nota: O modelo deve ser exportado pelo notebook como 'best_model.joblib'
COPY best_model.joblib* ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
