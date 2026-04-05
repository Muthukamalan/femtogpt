from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

# Global variables (shared across requests)
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        login(token=hf_token)

    global model, tokenizer

    # 🔹 Startup: load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded!")

    yield  # App runs here

    print("Shutting down...")
    model = None
    tokenizer = None


app = FastAPI(lifespan=lifespan)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)



class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40
    early_stopping:bool =True
    num_beams:int=4
    do_sample:bool=True


@app.post("/text/generate")
async def generate_text(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Encode the prompt
        prompt_ids = tokenizer.encode(request.prompt, return_tensors="pt",truncation=True,padding=True,padding_side='right')

        # Move to device if needed
        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )


        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0].tolist())

        return {"output_text": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}