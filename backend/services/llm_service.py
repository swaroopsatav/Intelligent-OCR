import hashlib
import time
from collections import OrderedDict

import torch

from backend.config.settings import (
    LLM_CACHE_SIZE,
    LLM_MAX_INPUT_TOKENS
)
from backend.models.qwen_loader import (
    QwenLoader
)


class LLMService:

    _model = None
    _tokenizer = None
    _cache = OrderedDict()

    @classmethod
    def load_model(cls):

        if cls._model is None:

            cls._model, cls._tokenizer = (
                QwenLoader.load_model()
            )

        return (
            cls._model,
            cls._tokenizer
        )

    @classmethod
    def _cache_key(
        cls,
        prompt,
        max_new_tokens
    ):
        raw_key = f"{max_new_tokens}\n{prompt}"

        return hashlib.sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

    @classmethod
    def _get_cached_response(
        cls,
        cache_key
    ):
        if cache_key not in cls._cache:
            return None

        response = cls._cache.pop(
            cache_key
        )
        cls._cache[cache_key] = response

        return response

    @classmethod
    def _set_cached_response(
        cls,
        cache_key,
        response
    ):
        if LLM_CACHE_SIZE <= 0:
            return

        cls._cache[cache_key] = response

        while len(cls._cache) > LLM_CACHE_SIZE:
            cls._cache.popitem(
                last=False
            )

    @classmethod
    def generate(
        cls,
        prompt,
        max_new_tokens=256
    ):

        cache_key = cls._cache_key(
            prompt,
            max_new_tokens
        )
        cached_response = cls._get_cached_response(
            cache_key
        )

        if cached_response is not None:
            return cached_response

        model, tokenizer = (
            cls.load_model()
        )

        messages = [
            {
                "role": "system",
                "content": """
You are a document understanding assistant.

Rules:
- Follow instructions exactly.
- When asked for JSON, return ONLY valid JSON.
- Do not add explanations.
- Do not add markdown.
- Do not add code fences.
- Do not add commentary.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=LLM_MAX_INPUT_TOKENS
        ).to(model.device)

        start_time = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                repetition_penalty=1.03,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        processing_time = round(
            time.time() - start_time,
            2
        )

        print(
            f"\nLLM Processing Time: "
            f"{processing_time} sec"
        )

        cls._set_cached_response(
            cache_key,
            response
        )

        return response
