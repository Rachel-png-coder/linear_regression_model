FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create models directory
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "8000"]