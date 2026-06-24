import time

from backend.models.qwen_loader import (
    QwenLoader
)


class LLMService:

    _model = None
    _tokenizer = None

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
    def generate(
        cls,
        prompt,
        max_new_tokens=256
    ):

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
            max_length=2048
        ).to(model.device)

        start_time = time.time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
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

        print(
            "\n===== LLM RESPONSE =====\n"
        )

        print(
            response
        )

        print(
            "\n========================\n"
        )

        return response
