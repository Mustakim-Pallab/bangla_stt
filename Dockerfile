FROM ai4bharat_model

RUN mkdir -p /ai4bharat
WORKDIR /ai4bharat


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7002
CMD ["python", "ai_bharat.py"]
