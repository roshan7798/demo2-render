FROM python:3.10-slim

ENV TZ=Asia/Tehran

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
