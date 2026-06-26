"""
inference_engine/model_runner.py

Wraps model loading + batched generation behind a small interface, so the
server doesn't care whether it's talking to a real model or a stand-in.

Two implementations:
- StubModelRunner: instant, no GPU/network needed. Used by the test suite and
  for plumbing work (dashboard, batching tuning) where you don't want to wait
  on real inference every time you run a test.
- QwenModelRunner: loads Qwen2.5-0.5B-Instruct and does real batched
  generation. This is what actually makes the system an inference engine
  instead of infrastructure wrapped around nothing.

Swap which one the server uses via the MODEL_BACKEND env var (see server.py).
This mirrors the same idea as Week 3's stub-first approach: build and verify
the plumbing against something fast and fake, then swap in the real thing
without touching anything else.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class StubModelRunner:
    """No model loaded at all. Echoes the prompt back. Use for fast tests
    and for developing everything EXCEPT the model itself."""

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        return [f"[stub-echo] {p}" for p in prompts]


class QwenModelRunner:
    """Loads Qwen2.5-0.5B-Instruct once at construction time, and runs REAL
    batched generation on every call.

    Batching detail worth understanding (this is the part that actually
    trips people up): decoder-only models need LEFT padding for batched
    generation. Real generation continues from the LAST token of each
    sequence. If shorter prompts in a batch are padded on the RIGHT (the
    default for most tokenizers), the model ends up trying to "continue"
    from a padding token instead of the real last word -- which produces
    garbage for every prompt shorter than the longest one in the batch.
    Left-padding keeps every real prompt's last token in the same final
    column, so generation picks up from the right place for everyone.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info("Loading %s on %s (dtype=%s) ...", model_name, self.device, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for correct batched generation, see class docstring

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

        logger.info("Model loaded and ready.")

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: deterministic output, easier to debug than sampling for now
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]  # strip the prompt back off, keep only what's newly generated
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        ]


if __name__ == "__main__":
    # Manual check that the real model loads and batched generation actually
    # works -- run this BEFORE running the full server, to isolate model
    # problems from server/plumbing problems.
    #
    # Run with: python model_runner.py
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    runner = QwenModelRunner()
    prompts = [
        "What is the capital of France?",
        "Write a one-sentence summary of what a GPU does.",
    ]
    outputs = runner.generate_batch(prompts, max_new_tokens=40)
    for p, o in zip(prompts, outputs):
        print(f"\nPROMPT: {p}\nOUTPUT: {o}")