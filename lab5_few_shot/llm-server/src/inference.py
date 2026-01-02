import re
import time
from abc import ABC, abstractmethod

import torch
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

        inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)  # type: ignore

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
            )

        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore

        end_time = time.perf_counter()
        duration = end_time - start_time
        log.info(f"Inference completed in: {duration:.2f} seconds")
        log.debug("Raw model response:")
        log.debug("#" * 45)
        log.debug(raw_response)
        log.debug("#" * 45)

        try:
            model_response = self._extract_model_response(raw_response)
        except InvalidResponseFormatError as e:
            log.error(f"Failed to parse model response: {e}")
            error_msg = f"Failed to parse model response: {e}"
            return ApiResponse(error=error_msg)

        return ApiResponse(response=model_response)

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
