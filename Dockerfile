FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
PY

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8501

#ENTRYPOINT ["python", "-m", "src.inference.predict"]

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
