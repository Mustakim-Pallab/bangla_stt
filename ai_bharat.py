from typing import Any

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

app = FastAPI()


class TranscriptionRequest(BaseModel):
    audio: Any
    language: str = "bn"
    tokenizer: str = "rnnt"


@app.post("/transcribe")
async def transcribe_audio(
        request: TranscriptionRequest
):
    if isinstance(request.audio, list):
        print("Transcribing from list of audio samples")
        request.audio = np.array(request.audio, dtype=np.float32)
        print(request.audio)

    wav = torch.from_numpy(request.audio).unsqueeze(0).float()
    # torchaudio.save("output.wav", wav, sample_rate=16000)
    wav = torch.mean(wav, dim=0, keepdim=True)
    print(wav)

    # torchaudio.save("output_1.wav", wav, sample_rate=16000)

    print("Running inference...")
    import time
    start_time = time.time()
    transcription_ctc = model(wav, request.language, request.tokenizer)
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)
    print("CTC Transcription:", transcription_ctc)
    return {
        "text": transcription_ctc,
        "language": request.language
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7001)
