# ======================================================
#   SenseVoice speech emotion sidecar
# ======================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY ser_api.py /app/

EXPOSE 50000

ENV MODELSCOPE_CACHE=/models
ENV PYTHONUNBUFFERED=1
ENV SENSEVOICE_DEVICE=auto
ENV SENSEVOICE_MODEL=iic/SenseVoiceSmall

RUN mkdir -p /models

CMD ["uvicorn", "ser_api:app", "--host", "0.0.0.0", "--port", "50000"]
