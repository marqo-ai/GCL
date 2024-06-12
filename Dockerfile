FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    make \
    awscli \
    git \
    build-essential

RUN pip install --upgrade pip

WORKDIR /app/GCL
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "evals/eval_gs_v1.py"]
