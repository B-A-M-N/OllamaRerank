FROM python:3.10-slim

WORKDIR /app

COPY rerank_stack/requirements.txt /app/rerank_stack/requirements.txt
RUN pip install --no-cache-dir -r rerank_stack/requirements.txt

COPY rerank_stack /app/rerank_stack

ENV PYTHONPATH="/app/rerank_stack/src"
ENV RERANK_BIND_PUBLIC=1

EXPOSE 8000

CMD ["uvicorn", "rerank_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
