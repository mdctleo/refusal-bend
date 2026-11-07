from __future__ import annotations

import asyncio
import os
import threading
from itertools import cycle
from pathlib import Path

import torch
from tqdm import tqdm
from vllm_omni.entrypoints import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
import hashlib


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class Diffusion:
    """
    Drop-in replacement for the original Diffusion class using vLLM-Omni.

    Public API is preserved:
    - constructor signature is unchanged
    - `create_images(prompts, save_paths)` writes images to disk
    - `edit_images(prompts, image_paths, save_paths)` writes images to disk
    """

    def __init__(
        self,
        model_name,
        num_inference_steps=4,
        guidance_scale=0.0,
        batch_size=16,
        num_images_per_prompt=1,
        seed=42,
    ):
        self.model_name = model_name
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt
        self.seed = seed

        self.backend_name = "vllm-omni"
        self._closed = False

        self.engine = self._load_engine()
        self.pipe = self.engine

    def _load_engine(self):
        model_id, dtype = self._resolve_model_config()
        tensor_parallel_size = max(torch.cuda.device_count(), 1)

        engine = AsyncOmniDiffusion(
            model=model_id,
            batch_size=self.batch_size,
            dtype=dtype,
            trust_remote_code=True,
            num_gpus=tensor_parallel_size,
            parallel_config={"tensor_parallel_size": tensor_parallel_size},
            cache_backend=os.getenv("DIFFUSION_2_OMNI_CACHE_BACKEND", "none"),
            vae_use_slicing=self._env_flag("DIFFUSION_2_OMNI_VAE_SLICING", default=False),
            vae_use_tiling=self._env_flag("DIFFUSION_2_OMNI_VAE_TILING", default=False),
            enable_multithread_weight_load=True,
            num_weight_load_threads=int(os.getenv("DIFFUSION_2_OMNI_WEIGHT_LOAD_THREADS", "4")),
            enforce_eager=self._env_flag("DIFFUSION_2_OMNI_ENFORCE_EAGER", default=False),
        )

        print(
            f"Initialized {self.model_name} with {self.backend_name}"
            f" using model={model_id}, dtype={dtype}, tp={tensor_parallel_size}",
            flush=True,
        )
        return engine

    def _resolve_model_config(self):
        if self.model_name == "flux2-dev":
            return ("black-forest-labs/FLUX.2-dev", "float16")
        if self.model_name == "flux2-klein-fast":
            return ("black-forest-labs/FLUX.2-klein-4B", "bfloat16")
        if self.model_name == "flux2-klein":
            return ("black-forest-labs/FLUX.2-klein-9B", "bfloat16")

        raise ValueError(f"Unknown model name: {self.model_name}")

    def _env_flag(self, name, default=False):
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def _runner():
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover - defensive bridge for async contexts
                error["value"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]
        return result.get("value")

    def _generate_numbered_paths(self, save_path, count):
        """Generate numbered file paths when multiple images are produced."""
        if count == 1:
            return [save_path]
        
        path = Path(save_path)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        
        return [str(parent / f"{stem}_{i}{suffix}") for i in range(count)]

    def _save_images(self, images, save_paths):
        for image, save_path in zip(images, save_paths):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)

    def _derive_seed(self, key):
        """Per-output seed: a stable hash of the base seed and the output's key, so the same
        key always renders the same image and distinct keys render distinct ones."""
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (2**31)

    def _build_sampling_params(self, *, seed_key, height=None, width=None, max_sequence_length=None):
        return OmniDiffusionSamplingParams(
            seed=self._derive_seed(seed_key),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_outputs_per_prompt=self.num_images_per_prompt,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
        )

    def _normalize_prompt_item(self, item):
        if isinstance(item, dict):
            prompt = item["prompt"]
            negative_prompt = item.get("negative_prompt")
        else:
            prompt = item
            negative_prompt = None
        return prompt, negative_prompt

    def _build_text_prompt_payload(self, item):
        prompt, negative_prompt = self._normalize_prompt_item(item)

        payload = {"prompt": prompt}
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        return payload

    def _build_edit_prompt_payload(self, prompt_item, image_path):
        prompt, negative_prompt = self._normalize_prompt_item(prompt_item)

        payload = {
            "prompt": prompt,
            "multi_modal_data": {"image": str(Path(image_path))},
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        return payload

    def create_images(self, prompts, save_paths, seed_keys=None):
        if len(prompts) != len(save_paths):
            raise ValueError("The number of prompts must match the number of save paths.")

        seed_keys = self._resolve_seed_keys(seed_keys, len(prompts))

        # vLLM-Omni's installed Flux2 pipelines currently process one prompt per request.
        for prompt_item, save_path, seed_key in tqdm(
            zip(prompts, save_paths, seed_keys),
            desc="Generating images...",
            total=len(prompts),
        ):
            sampling_params = self._build_sampling_params(
                seed_key=seed_key, height=1024, width=1024, max_sequence_length=256
            )
            prompt_payload = self._build_text_prompt_payload(prompt_item)
            output = self._run_async(self.engine.generate(prompt_payload, sampling_params))
            
            # Generate numbered paths if multiple images per prompt
            save_targets = self._generate_numbered_paths(save_path, len(output.images))
            self._save_images(output.images, save_targets)

    def _resolve_seed_keys(self, seed_keys, count):
        if seed_keys is None:
            return [str(i) for i in range(count)]
        if len(seed_keys) != count:
            raise ValueError("The number of seed keys must match the number of prompts.")
        return list(seed_keys)

    def edit_images(self, prompts, image_paths, save_paths, seed_keys=None):
        """
        Args:
            prompts (list of str): A list of text prompts describing the desired edits.
            image_paths (list of str): A list of image paths.
            save_paths (list of str): A list of file paths to save the edited images.
            seed_keys (list of str, optional): One key per output that fixes its seed together
                with the base seed. Defaults to the output's position in the list.

        Returns:
            None. Images are written to `save_paths`.
        """
        if len(prompts) != len(image_paths):
            raise ValueError("The number of prompts must match the number of images.")

        if len(image_paths) != len(save_paths):
            raise ValueError("The number of image paths must match the number of save paths.")

        seed_keys = self._resolve_seed_keys(seed_keys, len(prompts))

        for prompt_item, image_path, save_path, seed_key in tqdm(
            zip(prompts, image_paths, save_paths, seed_keys),
            desc="Editing images in batches...",
            total=len(prompts),
        ):
            sampling_params = self._build_sampling_params(seed_key=seed_key)
            prompt_payload = self._build_edit_prompt_payload(prompt_item, image_path)
            output = self._run_async(self.engine.generate(prompt_payload, sampling_params))
            edited_images = output.images

            # Generate numbered paths if multiple images per prompt
            if len(edited_images) > 1:
                save_targets = self._generate_numbered_paths(save_path, len(edited_images))
            else:
                save_targets = [save_path]

            self._save_images(edited_images, save_targets)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.engine is not None:
            self.engine.close()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
