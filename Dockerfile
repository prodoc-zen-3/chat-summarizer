FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime files (intentionally exclude test.py)
COPY main.py /app/
COPY command_permissions.json /app/

CMD ["python", "main.py"]
