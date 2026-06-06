FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --- CPU-only PyTorch (default):
# CLIP и cross-encoder инференсятся на CPU, поэтому GPU-runtime в контейнере не нужен.
# Ставим torch ДО requirements.txt, чтобы pip не подтянул GPU-вариант как dependency.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt

# --- GPU-вариант (раскомментируй и закомментируй CPU-блок выше): ---
# RUN pip install --no-cache-dir -r requirements.txt
# torch сам подтянется в GPU-сборке с nvidia-cuda-* зависимостями (~5 ГБ дополнительно)

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
