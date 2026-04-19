# Minimal container for the pair-level inference CLI.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/opt/torch-cache

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2').eval()"

COPY . ./

ENTRYPOINT ["python", "scripts/infer_pair.py"]
CMD ["--help"]
