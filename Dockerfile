FROM python:3.9.4-slim
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENTRYPOINT streamlit run main.py --server.address=0.0.0.0 --server.port=8080