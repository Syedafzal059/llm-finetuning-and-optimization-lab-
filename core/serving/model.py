"""vLLM model loader with LoRA/QLoRA adapter support."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Generator, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from core.utils.config_loader import resolve_path

logger = logging.getLogger(__name__)


class ServingModel:
    """Loads a base causal LM with optional LoRA adapter via vLLM."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with serving configuration."""
        self._config = config
        self._llm: Optional[LLM] = None
        self._lora_request: Optional[LoRARequest] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_display_name: str = ""
        self._base_model_name: str = ""
        self._adapter_path: Optional[str] = None
        self._loaded: bool = False
        self._generation_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has been loaded successfully."""
        return self._loaded and self._llm is not None

    @property
    def model_display_name(self) -> str:
        """Return a human-readable model identifier for responses."""
        return self._model_display_name

    @property
    def base_model_name(self) -> str:
        """Return the configured base model name."""
        return self._base_model_name

    @property
    def adapter_path(self) -> Optional[str]:
        """Return the resolved adapter path, if configured."""
        return self._adapter_path

    def load(self) -> None:
        """Load base model and adapter once at startup."""
        model_cfg = self._config["model"]
        self._base_model_name = model_cfg["model_name"]
        adapter_rel = model_cfg.get("adapter_path")

        if not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available. vLLM GPU serving requires a CUDA-capable device."
            )

        llm_kwargs: dict[str, Any] = {
            "model": self._base_model_name,
            "max_model_len": model_cfg.get("max_model_len", 4096),
            "gpu_memory_utilization": model_cfg.get("gpu_memory_utilization", 0.85),
            "dtype": model_cfg.get("dtype", "auto"),
            "trust_remote_code": model_cfg.get("trust_remote_code", False),
        }

        if model_cfg.get("quantization_enabled", True):
            llm_kwargs["quantization"] = model_cfg.get("quantization", "bitsandbytes")

        if adapter_rel:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = model_cfg.get("max_lora_rank", 64)

        logger.info("Loading vLLM model: %s", self._base_model_name)
        self._llm = LLM(**llm_kwargs)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._base_model_name,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )

        if adapter_rel:
            resolved = str(resolve_path(adapter_rel))
            if not Path(resolved).exists():
                logger.warning("Adapter path does not exist yet: %s", resolved)
            self._adapter_path = resolved
            self._lora_request = LoRARequest("adapter", 1, resolved)
            adapter_label = Path(adapter_rel).name.rstrip("/") or "adapter"
            self._model_display_name = (
                f"{self._short_model_name(self._base_model_name)} + {adapter_label}-adapter"
            )
        else:
            self._adapter_path = None
            self._model_display_name = self._short_model_name(self._base_model_name)

        self._loaded = True
        logger.info("Model loaded: %s", self._model_display_name)

    def count_prompt_tokens(self, prompt: str) -> int:
        """Return the token count for a prompt string."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized. Load the model first.")
        return len(self._tokenizer.encode(prompt, add_special_tokens=True))

    def get_fine_tune_status(self) -> dict[str, Any]:
        """Return model and adapter readiness details for /fine-tune-status."""
        adapter_path = Path(self._adapter_path) if self._adapter_path else None
        adapter_on_disk = adapter_path.is_dir() if adapter_path else False
        adapter_config_present = (
            (adapter_path / "adapter_config.json").is_file() if adapter_path else False
        )

        if self.is_loaded:
            readiness = "ok"
        elif self._loaded:
            readiness = "degraded"
        else:
            readiness = "not_ready"

        return {
            "status": readiness,
            "model": self._base_model_name or "unknown",
            "adapter": self._project_adapter_label(),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "adapter_on_disk": adapter_on_disk,
            "adapter_config_present": adapter_config_present,
            "model_loaded": self.is_loaded,
            "gpu_available": torch.cuda.is_available(),
        }

    def _project_adapter_label(self) -> Optional[str]:
        """Return a short adapter label derived from the configured path."""
        if not self._adapter_path:
            return None
        return Path(self._adapter_path).name or "adapter"

    def count_output_tokens(self, text: str) -> int:
        """Return the token count for generated output text."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized. Load the model first.")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, int]:
        """Run synchronous generation and return output text plus token counts."""
        if self._llm is None:
            raise RuntimeError("Model is not loaded.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        with self._generation_lock:
            return self._run_generate(prompt, sampling_params)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        """Yield incremental text deltas for streaming responses."""
        if self._llm is None:
            raise RuntimeError("Model is not loaded.")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        with self._generation_lock:
            engine = getattr(self._llm, "llm_engine", None)
            if engine is None:
                output_text, _, _ = self._run_generate(prompt, sampling_params)
                yield output_text
                return

            try:
                from vllm.utils import random_uuid
            except ImportError:
                from uuid import uuid4

                random_uuid = uuid4  # type: ignore[assignment]

            request_id = random_uuid()
            engine.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=self._lora_request,
            )

            previous_text = ""
            while engine.has_unfinished_requests():
                step_outputs = engine.step()
                for output in step_outputs:
                    if output.request_id != request_id:
                        continue
                    current_text = output.outputs[0].text
                    delta = current_text[len(previous_text):]
                    if delta:
                        yield delta
                    previous_text = current_text

    def _run_generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> tuple[str, int, int]:
        """Run a single vLLM generate call without acquiring the lock."""
        outputs = self._llm.generate(
            [prompt],
            sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        generated = outputs[0].outputs[0]
        tokens_in = len(outputs[0].prompt_token_ids)
        tokens_out = len(generated.token_ids)
        return generated.text, tokens_in, tokens_out

    @staticmethod
    def _short_model_name(model_name: str) -> str:
        """Convert a Hub model id to a compact display label."""
        return model_name.split("/")[-1].replace("_", "-")

    @staticmethod
    def get_gpu_memory_mb() -> tuple[float, float]:
        """Return (used_mb, total_mb) for the primary CUDA device."""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        used = torch.cuda.memory_allocated(0) / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        return round(used, 2), round(total, 2)

    @staticmethod
    def format_gpu_memory_gb(used_mb: float, total_mb: float) -> tuple[str, str]:
        """Format GPU memory values as human-readable GB strings."""
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024
        return f"{used_gb:.1f}GB", f"{total_gb:.0f}GB"

    @staticmethod
    def get_gpu_name() -> Optional[str]:
        """Return the CUDA device name when available."""
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
