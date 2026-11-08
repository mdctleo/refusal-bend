import pandas as pd
import os
import glob
from tqdm import tqdm
from filelock import FileLock
import json
import time
from system_prompts_2 import robot_constitution_64_automerge, robot_constitution_64_automerge_cosmos_reason2
from query_litellm import complete, target_params
from argparse import ArgumentParser


DEFAULT_MODEL = "gemini/gemini-robotics-er-2-preview"


class PlanGenerator():
    """Queries the target planner on each task image and writes Response / Thinking back to the CSV.

    `model` is a LiteLLM model string; its prefix selects the provider and the decoding each
    provider receives is defined by `query_litellm.target_params`, printed once here.
    """

    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        print(f"Target: {model}", flush=True)
        print(f"Decoding: {json.dumps(target_params(model), default=str)}", flush=True)


    
    def expand_for_evaluation(self, attack_csv_path="./attack.csv", attack_image_root="./attack_images/", output_path="./expanded_evaluation.csv"):
        df = pd.read_csv(attack_csv_path)

        expanded_rows = []

        print(len(df), flush=True)
        print("Columns in DataFrame: ", df.columns.tolist(), flush=True)
        print("Data types in DataFrame: ", df.dtypes, flush=True)

        for task in tqdm(df.to_dict("records"), total=len(df), desc="Expanding rows for evaluation"):
            attack_image_pattern = os.path.join(attack_image_root, f"{task['Task ID']}/{task['Object ID']}/*.png")
            attack_image_paths = sorted(glob.glob(attack_image_pattern))

            if not attack_image_paths:
                continue

            for attack_image_path in attack_image_paths:
                expanded_row = dict(task)
                image_name = os.path.basename(attack_image_path)
                expanded_row["Attack Image Path"] = f"{task['Task ID']}/{task['Object ID']}/{image_name}"
                expanded_rows.append(expanded_row)

        expanded_df = pd.DataFrame(expanded_rows)
        expanded_df.to_csv(output_path, index=False)
        print(f"Expanded evaluation CSV saved to {output_path}.", flush=True)
        return expanded_df

    def generate_plans(self, input_csv_path, output_csv_path: str, image_root: str, rang=None, is_attack=False, system_prompt_name="robot_constitution_64_automerge"):
        df = pd.read_csv(input_csv_path)

        if rang is not None:
            start, end = rang
            df = df.iloc[start:end]

        system_prompts = {
            "robot_constitution_64_automerge": robot_constitution_64_automerge,
            "robot_constitution_64_automerge_cosmos_reason2": robot_constitution_64_automerge_cosmos_reason2,
        }
        if system_prompt_name not in system_prompts:
            raise ValueError(f"Unknown system prompt {system_prompt_name!r}; expected one of {sorted(system_prompts)}.")
        system_prompt = system_prompts[system_prompt_name]()

        responses = []
        for idx, row in tqdm(df.iterrows(), desc="Generating plans", total=len(df)):
            task_id = row["Task ID"]
            instruction = row["Instruction"]

            if is_attack:
                pattern = os.path.join(image_root, f"{row['Attack Image Path']}")
            else:
                pattern = os.path.join(image_root, f"{row['Matched Image Path']}")

            print("Looking for images with pattern: ", pattern, flush=True)
            image_paths = glob.glob(pattern)
            print("Image paths found for task_id {}: {}".format(task_id, image_paths), flush=True)

            if not image_paths:
                print(f"Skipping task {task_id} due to missing images.", flush=True)
                continue

            for image_path in image_paths:
                result = self.query_target(
                    user_text=instruction,
                    image_path=image_path,
                    system_prompt=system_prompt
                )
                responses.append(result)

        self.write_results_to_csv(output_csv_path, responses, rang)

    def write_results_to_csv(self, output_path, responses, rang=None):
        """
        Write responses and thinking traces to CSV with file locking.
        Acquires lock, reads latest CSV, updates Response and Thinking columns, and writes back.
        
        Parameters:
            output_path (str): Path to the output CSV file.
            responses (list): List of response dicts with 'thinking' and 'text' keys.
            rang (tuple): Optional (start, end) range for partial updates.
        """
        lock_path = output_path + ".lock"

        with FileLock(lock_path):
            df = pd.read_csv(output_path)
            
            # Ensure Response and Thinking columns exist and are string type
            if "Response" not in df.columns:
                df["Response"] = ""
            if "Thinking" not in df.columns:
                df["Thinking"] = ""
            df["Response"] = df["Response"].fillna("").astype(str)
            df["Thinking"] = df["Thinking"].fillna("").astype(str)
            
            # If a range is specified, update only those rows
            if rang is not None:
                start, end = rang
                # Get the actual index labels for the range
                index_range = df.index[start:end]
                
                # Assign responses to the corresponding rows using their index labels
                assert len(responses) == len(index_range), "Number of responses does not match the number of rows in the specified range."
                for i, response in enumerate(responses):
                    if i < len(index_range):
                        # Handle both dict format (with thinking) and string format (backwards compat)
                        if isinstance(response, dict):
                            df.loc[index_range[i], "Response"] = response.get("text", "")
                            df.loc[index_range[i], "Thinking"] = response.get("thinking", "")
                        else:
                            df.loc[index_range[i], "Response"] = response
            
            # Write the updated DataFrame back to the CSV
            df.to_csv(output_path, index=False)


    def query_target(
        self,
        user_text: str,
        image_path: str,
        system_prompt: str = "",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> dict:
        """One planner query on `image_path` with `user_text` under `system_prompt`.

        Retries up to `max_retries` times on any error. Returns {"thinking": ..., "text": ...};
        both empty strings if every attempt fails.
        """
        for attempt in range(max_retries):
            try:
                result = complete(self.model, system_prompt, user_text, image_path)
                print("Response:", result["text"], flush=True)
                if result["thinking"]:
                    print("Thinking trace:", result["thinking"], flush=True)
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}", flush=True)
                time.sleep(retry_delay)

        print(f"All {max_retries} attempts failed. Returning empty string.", flush=True)
        return {"thinking": "", "text": ""}


if __name__ == "__main__":
    parser = ArgumentParser(description="Query the target planner on every task image of a CSV.")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Path to the input CSV file containing tasks.")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to the output CSV file to save results.")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory where images are stored.")
    parser.add_argument("--attack_image_root", type=str, required=False, help="Root directory where attacked images are.")
    parser.add_argument("--rang", type=str, default=None, help="Range of rows to process in the format 'start,end'.")
    parser.add_argument("--is_attack", action='store_true', help="Whether the input CSV is for attack tasks (uses Attack Image Path).")
    parser.add_argument("--is_expand", action='store_true', help="Whether to run the expansion step for evaluation (generates Attack Image Path).")
    parser.add_argument("--system_prompt_name", type=str, default="robot_constitution_64_automerge", help="Name of the system prompt to use.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Target planner as a LiteLLM model string, e.g. gemini/gemini-robotics-er-2-preview, "
                             "anthropic/claude-opus-5-5, openai/gpt-5.4-1, hosted_vllm/nvidia/Cosmos3-Nano.")

    args = parser.parse_args()
    if args.rang is not None:
        rang = tuple(map(int, args.rang.split(",")))
    else:
        rang = None

    plan_generator = PlanGenerator(model=args.model)

    if args.is_expand:
        plan_generator.expand_for_evaluation(
            attack_csv_path=args.input_csv_path,
            attack_image_root=args.image_root,
            output_path=args.output_csv_path
        )
    else:
        plan_generator.generate_plans(
            input_csv_path=args.input_csv_path,
            output_csv_path=args.output_csv_path,
            image_root=args.image_root,
            rang=rang,
            is_attack=args.is_attack,
            system_prompt_name=args.system_prompt_name
        )
