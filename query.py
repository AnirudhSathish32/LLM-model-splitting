from dataclasses import dataclass
import torch


@dataclass
class Query:
    prompt: str
    model_name: str
    session_id: str
    tokens_to_generate: int
    dtype: str

    @property
    def dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype, torch.float16)
    