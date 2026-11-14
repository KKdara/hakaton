FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False
COPY requirements-web.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt -r requirements.txt
COPY . .
RUN useradd --create-home app && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
