import io
from typing import Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

app = FastAPI()


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    language: str = Form("bn"),
):
    # Read uploaded file
    contents = await file.read()

    # Load audio into torch tensor
    wav, sr = torchaudio.load(io.BytesIO(contents))

    # Convert stereo → mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if needed
    target_sample_rate = 16000
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)

    # Run inference (CTC)
    transcription_ctc = model(wav, language, "ctc")

    # Run inference (RNNT)
    transcription_rnnt = model(wav, language, "rnnt")

    return {
        "language": language,
        "ctc_transcription": transcription_ctc,
        "rnnt_transcription": transcription_rnnt,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7002)
