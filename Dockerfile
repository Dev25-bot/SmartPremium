# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps for xgboost if needed
RUN apt-get update && apt-get install -y build-essential git --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Expose Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
