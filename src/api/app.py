"""
FastAPI REST inference microservice for Nebium chess move prediction.
"""

import os
import hydra
import torch
from fastapi import FastAPI
from hydra import compose, initialize
from hydra.utils import instantiate
from pydantic import BaseModel

from src.data.prepare import get_tokenizer

app = FastAPI(title="Nebium Chess API")

# Global state
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global MODEL, TOKENIZER
    # Setup Hydra
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../../configs")
    cfg = compose(config_name="config", overrides=["model=nebium_stub"])
    
    TOKENIZER = get_tokenizer(cfg)
    MODEL = instantiate(cfg.model)
    
    ckpt_path = "best_model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(ckpt["model_state_dict"])
        
    MODEL.to(DEVICE)
    MODEL.eval()

class PredictRequest(BaseModel):
    moves: str = ""
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    prompt_str = req.moves.strip()
    if prompt_str:
        encoded = TOKENIZER.encode(prompt_str)
        input_ids = [TOKENIZER.bos_id] + encoded
    else:
        input_ids = [TOKENIZER.bos_id]
        
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    mask = torch.ones_like(x, dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        output = MODEL.generate(
            x, mask,
            max_new_tokens=1,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )
        
    next_token_id = output[0][-1].item()
    if next_token_id in (TOKENIZER.eos_id, TOKENIZER.pad_id):
        predicted_move = ""
    else:
        token_str = TOKENIZER.decode([next_token_id]).strip()
        predicted_move = token_str.split()[0] if token_str else ""
        
    return {"predicted_move": predicted_move}
