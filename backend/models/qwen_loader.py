from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch

from backend.config.settings import (
    HF_LOCAL_FILES_ONLY,
    MODEL_NAME
)


class ModelLoadingError(RuntimeError):
    pass


class QwenLoader:

    tokenizer = None
    model = None

    @classmethod
    def load_model(cls):

        if cls.model is None:

            model_name = MODEL_NAME

            if not model_name:
                raise ModelLoadingError(
                    "MODEL_NAME is not configured."
                )

            print(
                f"Loading {model_name}..."
            )

            try:

                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                cls.tokenizer = (
                    AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        local_files_only=HF_LOCAL_FILES_ONLY
                    )
                )

                cls.model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=(
                            torch.float16
                            if torch.cuda.is_available()
                            else torch.float32
                        ),
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=HF_LOCAL_FILES_ONLY,
                        attn_implementation="sdpa"
                    )
                )

            except Exception as e:

                cls.tokenizer = None
                cls.model = None

                raise ModelLoadingError(
                    f"Unable to load model '{model_name}': {e}"
                ) from e

            print(
                "LLM Loaded Successfully"
            )

        return (
            cls.model,
            cls.tokenizer
        )
