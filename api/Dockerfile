FROM python:3.11-slim
WORKDIR /app
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r /app/ml/requirements.txt
COPY ml /app/ml
COPY api /app/api
COPY artifacts /app/artifacts
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
