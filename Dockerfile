FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
  curl \
  vim \
  git \
  make

WORKDIR /arc-vae

COPY . .

RUN pip install --no-cache-dir -U -r requirements.txt -e .
RUN python get_dataset.py
