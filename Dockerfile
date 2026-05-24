# Gunakan Python 3.10 (Versi paling stabil untuk TensorFlow)
FROM python:3.10-slim

# Set environment variables agar Python tidak membuat file .pyc dan tidak menahan log (unbuffered)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set direktori kerja di dalam container
WORKDIR /app

# Install sistem dependensi dasar jika pandas/numpy membutuhkannya
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements terlebih dahulu untuk caching layer Docker yang optimal
COPY requirements.txt .

# Install pustaka Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file proyek (api, Data, model .keras) ke dalam container
COPY . .

# Ekspos port 8000 yang akan digunakan oleh FastAPI
EXPOSE 8000

# Jalankan server FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
