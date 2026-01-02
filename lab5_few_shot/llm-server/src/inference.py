import re
import time
from abc import ABC, abstractmethod

import torch
from fastapi import HTTPException
from src.dto import ApiResponse, InferenceRequest
from src.logger import log
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import override


class ModelInferencePort(ABC):
    @abstractmethod
    def infer(self, request: InferenceRequest) -> ApiResponse:
        pass


class InvalidResponseFormatError(Exception):
    pass


class ModelInferenceAdapter(ModelInferencePort):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device

    @override
    def infer(self, request: InferenceRequest) -> ApiResponse:
        log.debug(f"Starting inference for request: {request}")

        try:
            messages = [{"role": "user", "content": request.prompt}]
            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)  # type: ignore

            start_time = time.perf_counter()
            inputs = self.tokenizer.apply_chat_template(  # type: ignore
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(  # type: ignore
                    input_ids=inputs,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,  # type: ignore
                    pad_token_id=self.tokenizer.eos_token_id,  # type: ignore
                    use_cache=False,
                )

            input_length = inputs.shape[-1]
            generated_tokens = outputs[0][input_length:]

            full_resposne = self.tokenizer.decode(  # type: ignore
                outputs[0], skip_special_tokens=True
            )

            raw_response = self.tokenizer.decode(  # type: ignore
                generated_tokens, skip_special_tokens=True
            )

            end_time = time.perf_counter()
            duration = end_time - start_time
            log.info(f"Inference completed in: {duration:.2f} seconds")
            log.debug("Raw model response:")
            log.debug("#" * 45)
            log.debug(full_resposne)
            log.debug("#" * 45)

            return ApiResponse(response=raw_response)
        except Exception as e:
            log.error(f"Inference failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error during inference: {str(e)}"
            )

    @staticmethod
    def _extract_model_response(text: str) -> str:
        if "__RESPONSE__:" in text:
            text = text.split("__RESPONSE__:", 1)[1]

        text = text.strip()

        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if match:
            return match.group(0)

        if "return" not in text:
            raise InvalidResponseFormatError(
                "No 'return' statement found in the response"
            )

        raise InvalidResponseFormatError("No valid JSON array found in the response")
