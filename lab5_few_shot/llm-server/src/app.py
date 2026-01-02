from typing import Tuple

import torch
from fastapi import FastAPI
from src.config import Config
from src.dto import ApiResponse, InferenceRequest
from src.inference import ModelInferenceAdapter, ModelInferencePort
from src.logger import log
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_app(config: Config) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Starting model loading...")
    log.info(f"Device: {device}")
    model, tokenizer = _load_model(config)

    model.eval()  # type: ignore
    model.to(device)  # type: ignore[attr-defined]
    log.info("Model loading completed.")

    inference_port: ModelInferencePort = ModelInferenceAdapter(
        model=model,  # type: ignore
        tokenizer=tokenizer,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    app = FastAPI(title="LLM Server", version="1.0.0")

    @app.post("/generate")
    async def generate(request: InferenceRequest) -> ApiResponse:
        inference_repsonse = inference_port.infer(request)

        return inference_repsonse

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


def _load_model(config: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    log.info(f"Loading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    log.info(f"Model {config.model_name} loaded successfully")
    return model, tokenizer  # type: ignore[return-value]
