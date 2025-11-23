FROM python:3.11-slim

WORKDIR /app

COPY requirments.txt .

RUN pip install --no-cache-dir -r requirments.txt

COPY . .

EXPOSE 8000

CMD ["streamlit" , "run" , "frontend.py"]