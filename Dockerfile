
# ML 서비스 Dockerfile 예시
FROM python:3.10.16
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY app/data.csv /app/data.csv
EXPOSE 5000
CMD ["python", "/app/app/main.py"]
