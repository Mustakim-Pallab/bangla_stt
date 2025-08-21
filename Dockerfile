FROM mustakimpallab/python-ai-starter

RUN mkdir -p /ai4bharat
WORKDIR /ai4bharat


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('ai4bharat/indic-conformer-600m-multilingual', trust_remote_code=True)"

COPY . .

EXPOSE 7001
CMD ["python", "ai_bharat.py"]
