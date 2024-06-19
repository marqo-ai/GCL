FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    make \
    build-essential

RUN pip install --upgrade pip

WORKDIR /app/GCL
COPY evals evals
COPY requirements.txt .

RUN pip install -r requirements.txt
ENTRYPOINT ["python", "evals/eval_gs_v1.py"]
