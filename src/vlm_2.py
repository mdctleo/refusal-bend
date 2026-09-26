import copy
import gc
import json
import os
from pathlib import Path
from urllib.parse import unquote

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor
from vllm import LLM as VLLMEngine, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class VLM:
    def __init__(
        self,
        model_name,
        system_prompt,
        max_new_tokens=500,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
        seed=42,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = int(top_k)
        self.seed = seed
        self.thinking_budget = None

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._engine_sleeping = False

        print(f"Model name: {self.model_name}", flush=True)

        if self._is_gemini_model():
            print("Initializing Gemini model client...", flush=True)
            self.model = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
            return

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is not None:
            self.tokenizer.padding_side = "left"

        self._initialize_local_model()

    def _is_gemini_model(self):
        return self.model_name.startswith("gemini")

    def _is_qwen3_family(self):
        return "Qwen3" in self.model_name

    def _initialize_local_model(self):
        tensor_parallel_size = max(torch.cuda.device_count(), 1)
        self.model = VLLMEngine(
            model=self.model_name,
            # max_model_len=32768,
            max_model_len=16384,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            seed=self.seed,
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 2},
            allowed_local_media_path=str(Path.cwd()),
        )
        self._engine_sleeping = False
    
    def move_to_cpu(self):
        if self._is_gemini_model():
            print("Gemini client does not use local GPU memory.", flush=True)
            return

        if self.model is None:
            print("No model is currently loaded.", flush=True)
            return

        print(
            f"GPU memory allocated before move: {torch.cuda.memory_allocated() / 1e6:.2f} MB",
            flush=True,
        )
        print(
            f"GPU memory reserved before move: {torch.cuda.memory_reserved() / 1e6:.2f} MB",
            flush=True,
        )

        try:
            self.model.sleep(level=1)
        except Exception as exc:
            print(
                f"Level-1 vLLM sleep failed ({exc}), falling back to level-2 sleep.",
                flush=True,
            )
            self.model.sleep(level=2)
        self._engine_sleeping = True
        torch.cuda.empty_cache()
        gc.collect()

        print(
            f"GPU memory allocated after move: {torch.cuda.memory_allocated() / 1e6:.2f} MB",
            flush=True,
        )
        print(
            f"GPU memory reserved after move: {torch.cuda.memory_reserved() / 1e6:.2f} MB",
            flush=True,
        )
        print(
            f"Model {self.model_name} moved to CPU and GPU memory cleared.",
            flush=True,
        )

    def move_to_gpu(self):
        if self._is_gemini_model():
            print("Gemini client does not use a local GPU.", flush=True)
            return

        if self.model is None:
            self._initialize_local_model()
            print(f"Model {self.model_name} moved back to GPU.", flush=True)
            return

        if self._engine_sleeping:
            self.model.wake_up()
            self._engine_sleeping = False
        print(f"Model {self.model_name} moved back to GPU.", flush=True)

    def _vllm_chat_template_kwargs(self, enable_thinking=True):
        if self._is_qwen3_family() and enable_thinking:
            return {"enable_thinking": True}
        elif self._is_qwen3_family() and not enable_thinking:
            return {"enable_thinking": False}
        
        return None

    def _extract_request_text(self, request_output):
        if not request_output.outputs:
            return ""
        # print("Extracted request output text: ", request_output.outputs[0].text, flush=True)
        return request_output.outputs[0].text
    
    def _extract_thinking_trace(self, request_output):
        """Extract thinking trace and response from request output text.
        
        Everything before </think> is thinking trace.
        Everything after </think> is the response text.
        
        Returns:
            tuple: (thinking_trace, response_text)
        """
        if not request_output.outputs:
            return "", ""
        output_text = request_output.outputs[0].text or ""
        end_tag = "</think>"
        end_idx = output_text.find(end_tag)
        
        if end_idx == -1:
            # No thinking tag found, return empty thinking trace and full text as response
            return "", output_text.strip()
        
        thinking_trace = output_text[:end_idx].strip()
        response_text = output_text[end_idx + len(end_tag):].strip()

        # print(f"Extracted thinking trace: '{thinking_trace}'", flush=True)
        # print(f"Extracted response text: '{response_text}'", flush=True)
        
        return thinking_trace, response_text

    def _parse_json_response(self, output_text, thinking_trace=None):
        try:
            result = json.loads(output_text)
            if thinking_trace is not None:
                result["thinking_trace"] = thinking_trace
            return result
        except json.JSONDecodeError:
            error_result = {"error": "Failed to parse JSON response", "raw_output": output_text}
            return error_result

    def _run_chat(self, conversations, sampling_params, add_generation_prompt=True, continue_final_message=False, enable_thinking=True):
        if not conversations:
            return []

        return self.model.chat(
            messages=conversations,
            sampling_params=sampling_params,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            use_tqdm=True,
            chat_template_kwargs=self._vllm_chat_template_kwargs(enable_thinking=enable_thinking),
        )
    
    def set_new_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def _build_user_content(self, prompt, image_paths=None):
        content = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            resolved_image_path = Path(image_path).expanduser().resolve()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": resolved_image_path.as_uri()},
                }
            )
        return content

    def _build_message(self, prompt, image_paths=None):
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self._build_user_content(prompt, image_paths=image_paths),
            },
        ]

    def _message_to_gemini_content(self, message):
        gemini_content = []
        for msg in message:
            if msg["role"] == "system":
                continue

            if msg["role"] != "user":
                continue

            for content in msg["content"]:
                if content["type"] == "text":
                    gemini_content.append(content["text"])
                elif content["type"] == "image_url":
                    image_url = content["image_url"]["url"]
                    image_path = (
                        Path(unquote(image_url.removeprefix("file://")))
                        if image_url.startswith("file://")
                        else Path(unquote(image_url))
                    )
                    with Image.open(image_path) as image:
                        gemini_content.append(image.convert("RGB"))
                elif content["type"] == "image":
                    with Image.open(content["image"]) as image:
                        gemini_content.append(image.convert("RGB"))

        return gemini_content

    def query_gemini(self, messages, response_schema):
        responses = []

        for message in tqdm(messages, desc="Querying Gemini", total=len(messages), leave=True):
            try:
                gemini_content = self._message_to_gemini_content(message)
                config_kwargs = {
                    "system_instruction": self.system_prompt,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "seed": self.seed,
                    "max_output_tokens": self.max_new_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                }
                if self.thinking_budget is not None:
                    config_kwargs["thinking_budget"] = self.thinking_budget

                response = self.model.models.generate_content(
                    model=self.model_name,
                    contents=gemini_content,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                if not response.text:
                    raise ValueError("Gemini model returned empty response")

                responses.append(json.loads(response.text))
            except Exception as exc:
                print(f"Error querying Gemini model: {exc}", flush=True)
                responses.append({"error": 1, "reason": str(exc)})

        return responses

    def query(self, messages, response_schema):
        if self._is_gemini_model():
            return self.query_gemini(messages, response_schema)
        return self.query_local_vlm(messages, response_schema=response_schema)
    
    def query_vlm_judge(
        self,
        prompts,
        original_image_paths,
        attacked_image_paths,
        response_schema=None,
    ):
        messages = []
        for prompt, original_image, attacked_image in zip(
            prompts,
            original_image_paths,
            attacked_image_paths,
        ):
            print("Original image path:", original_image, flush=True)
            print("Attacked image path:", attacked_image, flush=True)
            messages.append(
                self._build_message(
                    prompt,
                    image_paths=[original_image, attacked_image],
                )
            )

        return self.query(messages, response_schema=response_schema)

    def query_vlm(self, prompts, image_paths, response_schema=None):
        messages = []
        for prompt, image in zip(prompts, image_paths):
            print("Image path:", image, flush=True)
            messages.append(self._build_message(prompt, image_paths=[image]))

        return self.query(messages, response_schema=response_schema)
    
    def query_local_vlm(self, messages, response_schema=None):
        generation_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            max_tokens=self.max_new_tokens,
            structured_outputs=StructuredOutputsParams(json=response_schema) if response_schema else None,
        )

        request_outputs = self._run_chat(messages, generation_params)

        parsed_outputs = []
        for request_output in request_outputs:
            if request_output is None:
                parsed_outputs.append({"error": "Empty response"})
            else:
                thinking_trace, response_text = self._extract_thinking_trace(request_output)
                result = self._parse_json_response(response_text, thinking_trace=thinking_trace)

                parsed_outputs.append(result)

                # if response_schema:
                #     result = self._parse_json_response(response_text, thinking_trace=thinking_trace)
                #     parsed_outputs.append(result)
                # else:
                #     print("got here!", flush=True)
                #     parsed_outputs.append({
                #         "response": response_text,
                #         "thinking_trace": thinking_trace,
                #     })

        return parsed_outputs
    
    
    ################ LOGITS STUFF THAT I MAY NEED LATER ################
    def query_local_vlm_logits(self, prompts, image_paths):
        messages = []
        for prompt, image in zip(prompts, image_paths):
            # print("Image path:", image, flush=True)
            messages.append(self._build_message(prompt, image_paths=[image]))

        return self.query_local_vlm_logits_helper(messages=messages)
    
    def _sum_prompt_logprob_from(self, request_output):
        """Return the logprob of the first yes/no token after 'Final Verdict:'.
        (The name is a legacy misnomer: a single token's logprob is returned, not a sum.)
        
        Since 'Final Verdict: yes/no' is appended via continue_final_message, it will be
        in the last few tokens. Only examines the final 50 tokens for efficiency.
        """

        prompt_token_ids = request_output.prompt_token_ids or []
        prompt_logprobs = request_output.prompt_logprobs or []

        # Only look at the last 50 tokens since verdict is near the end
        start_idx = max(0, len(prompt_token_ids) - 100)
        # start_idx = 0  # Start from the beginning to ensure we capture all tokens for reconstruction
        
        # Reconstruct text from final tokens
        final_prompt = ""
        token_positions = {}  # Map character positions to token indices
        
        for pos in range(start_idx, len(prompt_token_ids)):
            tok = prompt_token_ids[pos]
            row = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            if row and tok in row:
                decoded_token = getattr(row[tok], 'decoded_token', '')
                # print(f"Position: {pos}, Token ID: {tok}, Logprobs Row: {row}, Decoded Token: {decoded_token}", flush=True)
                token_positions[len(final_prompt)] = pos  # Store where this token starts
                final_prompt += decoded_token
        
        # print(f"Final tokens text: '{final_prompt}'", flush=True)
        
        # Find where "Final Verdict:" ends
        verdict_start_pos = final_prompt.find("Final Verdict:")
        if verdict_start_pos == -1:
            return float("-inf")
        verdict_end_pos = verdict_start_pos + len("Final Verdict:")
        # print(f"'Final Verdict:' ends at character position {verdict_end_pos}", flush=True)
        
        # Find the token position corresponding to after "Final Verdict:"
        start_token_pos = None
        for char_pos in sorted(token_positions.keys()):
            if char_pos >= verdict_end_pos:
                start_token_pos = token_positions[char_pos]
                break
        
        if start_token_pos is None:
            return float("-inf")

        # Iterate forward from start_token_pos to find first yes/no token
        for pos in range(start_token_pos, len(prompt_token_ids)):
            tok = prompt_token_ids[pos]
            row = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            if row is None or tok not in row:
                continue
            
            logprob_entry = row[tok]
            decoded_token = getattr(logprob_entry, 'decoded_token', f"<token_{tok}>")
            
            # print(f"Position: {pos}, Token ID: {tok}, Decoded Token: '{decoded_token}', Logprob: {logprob_entry.logprob}", flush=True)
            
            # Check if this is a yes/no token
            decoded_lower = decoded_token.strip().lower()
            if decoded_lower in ["yes", "no"]:
                # print(f"Found first '{decoded_lower}' token after 'Final Verdict:' at position {pos} with Logprob: {logprob_entry}", flush=True)
                return logprob_entry.logprob
        
        return float("-inf")  # If no yes/no token found, return -inf logprob

    def _append_assistant_text(self, base_conversations, assistant_prefills):
        """Append assistant prefill text to existing conversations.
        
        Args:
            base_conversations: List of message lists (system/user context from pass 1)
            assistant_prefills: List of assistant prefill strings to continue from
            
        Returns:
            List of message lists with assistant text appended
        """
        final_messages = []
        for conversation, prefill in zip(base_conversations, assistant_prefills):
            messages = copy.deepcopy(conversation)
            messages.append(
                {
                    "role": "assistant",
                    "content": prefill,
                }
            )

            # print(f"Constructed message for scoring: {messages}", flush=True)
            final_messages.append(messages)
        return final_messages

    def score_candidate_batch(self, base_conversations, assistant_prefills):
        """Score assistant prefills by continuing the final assistant message and reading prompt logprobs."""
        final_messages = self._append_assistant_text(
            base_conversations,
            assistant_prefills,
        )
        outputs = self._run_chat(
            final_messages,
            SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,
                flat_logprobs=True,
                temperature=0.0,
                top_k=self.top_k,
                top_p=self.top_p,
                seed=self.seed,
            ),
            add_generation_prompt=False,
            continue_final_message=False,
            enable_thinking=False,
        )
        return [self._sum_prompt_logprob_from(o) for o in outputs]
    

    def query_local_vlm_logits_helper(self, messages):
        """Query model with thinking, then score yes/no continuations using logprobs."""
        # Generate thinking trace
        thought_outputs = self._run_chat(
            messages,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=2048,
                stop=["</think>"],
                include_stop_str_in_output=True,
                top_k=self.top_k,
                top_p=self.top_p,
                seed=self.seed,
            ),
            enable_thinking=True,
        )

        thought_traces = [self._extract_request_text(output) for output in thought_outputs]
        
        # Append verdict prefix to the assistant message (this will be the base for scoring)
        verdict_prefix_yes = "\nAnswer with exactly one word, either 'yes' or 'no'.\n\nFinal Verdict: yes"
        verdict_prefix_no = "\nAnswer with exactly one word, either 'yes' or 'no'.\n\nFinal Verdict: no"

        assistant_prefills_yes = [trace + verdict_prefix_yes for trace in thought_traces]
        assistant_prefills_no = [trace + verdict_prefix_no for trace in thought_traces]
        
        # Score " yes" and " no" continuations
        yes_sum_lps = self.score_candidate_batch(
            base_conversations=messages,
            assistant_prefills=assistant_prefills_yes,
        )
        no_sum_lps = self.score_candidate_batch(
            base_conversations=messages,
            assistant_prefills=assistant_prefills_no,
        )

        results = []
        for thought_trace, yes_lp, no_lp in zip(thought_traces, yes_sum_lps, no_sum_lps):
            logits = torch.tensor([yes_lp, no_lp], dtype=torch.float32, device="cpu")
            probs = torch.softmax(logits, dim=0)
            yes_p, no_p = float(probs[0]), float(probs[1])
            results.append(
                {
                    "thinking_trace": thought_trace,
                    "yes_lp": yes_lp,
                    "no_lp": no_lp,
                    "yes_p": yes_p,
                    "no_p": no_p,
                }
            )

        return results
