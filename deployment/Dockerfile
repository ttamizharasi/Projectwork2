FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "--timeout", "120", "-b", "0.0.0.0:10000", "app:app"]
