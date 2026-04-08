FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# HF Spaces requires exactly port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
