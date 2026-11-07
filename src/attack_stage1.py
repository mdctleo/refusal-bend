from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import os
from system_prompts_2 import image_instruction_scene, image_instruction_scene_user
from system_prompts_2 import scene_only_object, scene_only_object_user, RANDOM_OBJECT_LIST
import uuid
from argparse import ArgumentParser



class AttackStage1():
    def __init__(self, vlm_name):
        self.vlm_name = vlm_name
        self.vlm = None

    def filter_from_baseline_evaluation(self, input_csv_path, evaluation_csv_path):
        input_df = pd.read_csv(input_csv_path)
        evaluation_df = pd.read_csv(evaluation_csv_path)

        if "Judgement" not in evaluation_df.columns:
            print("No filtering done since 'Judgement' column is missing in the evaluation DataFrame.", flush=True)
            return input_df

        # Filter the evaluation DataFrame to keep only rejected images (Judgement == 0)
        rejected_evaluation = evaluation_df[evaluation_df["Judgement"] == 0]

        # Get the Task IDs of the rejected images
        rejected_task_ids = set(rejected_evaluation["Task ID"])

        # Filter the input DataFrame to keep only rows with Task IDs in the rejected set
        filtered_input_df = input_df[input_df["Task ID"].isin(rejected_task_ids)]

        # Just for testing, lets have only 10 samples
        # filtered_input_df = filtered_input_df.head(10)

        print("Length of filtered input DataFrame: ", len(filtered_input_df), flush=True)

        filtered_input_df = filtered_input_df.reset_index(drop=True)

        return filtered_input_df

    def generate_co_objects(self, df, output_csv_path, n=5, k=3, image_root=None, system_prompt_name="image_and_task_instruction"):
        """Propose n object sets of k objects per task.

        system_prompt_name selects the proposer (see analysis/rq2/README.md):
          image_and_task_instruction  RQ1: sees image + instruction, red-team prompt (default)
          scene_only                  RQ2: sees the image only, neutral prompt
          random_list                 RQ2: no VLM call; draws from RANDOM_OBJECT_LIST
          null_edit                   RQ2: no VLM call; n empty sets per task, so the editor adds nothing
        The RQ2 modes also write `Attack Image Path` as {Task ID}/{Object ID}/final_0.png,
        the name attack_original.py produces at --num_samples 1, so no expand step is needed.
        """
        assert system_prompt_name in ("image_and_task_instruction", "scene_only", "random_list", "null_edit"), system_prompt_name
        output= []

        unique_task_ids = df['Task ID'].unique()
        user_prompts = []
        image_paths = []
        system_prompt, response_schema = None, None
        for task_id in tqdm(unique_task_ids, total=len(unique_task_ids), desc="Generating co-object sets"):
            task_rows = df[df['Task ID'] == task_id]
           
            if "score" in df.columns:
                # Get the 'co_object_set' values for the current Task ID as an array
                co_object_sets = task_rows["co_object_set"].dropna().tolist()
                scores = task_rows["score"].dropna().tolist()

                assert len(co_object_sets) == len(scores), f"Length of co_object_sets ({len(co_object_sets)}) does not match length of scores ({len(scores)}) for Task ID {task_id}"

                # Get feedback if available
                feedbacks = []
                if "feedback" in task_rows.columns:
                    feedbacks = task_rows["feedback"].dropna().tolist()
                    assert len(feedbacks) == len(scores), f"Length of feedbacks ({len(feedbacks)}) does not match length of scores ({len(scores)}) for Task ID {task_id}"
                    # Parse feedback JSON strings into dictionaries
                    feedbacks = [json.loads(fb) if isinstance(fb, str) else fb for fb in feedbacks]
                else:
                    feedbacks = [None] * len(scores)

                # Build the dictionary with co_object_sets as keys and dicts with score and feedback as values
                co_object_sets = dict(zip(
                    co_object_sets,
                    [{"score": score, "feedback": feedback} for score, feedback in zip(scores, feedbacks)]
                ))
                print(f"Co-object sets, scores, and feedback for Task ID {task_id}: {co_object_sets}", flush=True)
            else:
                co_object_sets = None

            instruction = task_rows.iloc[0]["Instruction"]
            matched_image_path = task_rows.iloc[0]["Matched Image Path"]

            if system_prompt_name == "scene_only":
                user_prompts.append(scene_only_object_user())
                system_prompt, response_schema = scene_only_object(n=n, k=k)
            else:
                user_prompts.append(image_instruction_scene_user(task_instruction=instruction, previous_objects=co_object_sets))
                system_prompt, response_schema = image_instruction_scene(n=n, k=k)
            image_paths.append(os.path.join(image_root, matched_image_path))

        # Initialize response placeholders
        co_object_sets = [None] * len(df['Task ID'].unique())

        if system_prompt_name == "random_list":
            print(f"random_list mode: no proposer call; drawing {n*k} of {len(RANDOM_OBJECT_LIST)} list items per task", flush=True)
        elif system_prompt_name == "null_edit":
            print(f"null_edit mode: no proposer call; {n} empty object sets per task", flush=True)
        else:
            print("First user prompt: ", user_prompts[0], flush=True)

        if system_prompt_name == "null_edit":
            responses = [{"sets": [[] for _ in range(n)]} for _ in unique_task_ids]
        elif system_prompt_name == "random_list":
            # No proposer: draw n*k list items per task without replacement, one seed per task index.
            assert n * k <= len(RANDOM_OBJECT_LIST), f"n*k={n*k} exceeds the {len(RANDOM_OBJECT_LIST)}-item list"
            responses = []
            for task_index in range(len(unique_task_ids)):
                rng = np.random.default_rng(42 + task_index)
                draw = [RANDOM_OBJECT_LIST[i] for i in rng.choice(len(RANDOM_OBJECT_LIST), size=n * k, replace=False)]
                responses.append({"sets": [draw[j * k:(j + 1) * k] for j in range(n)]})
        else:
            responses = self.query_vlm(
                user_prompts=user_prompts,
                system_prompt=system_prompt,
                temperature=0.7,
                top_p=0.9,
                top_k=0.0,
                seed=42,
                image_paths=image_paths,
                max_new_tokens=8192,
                response_schema=response_schema
            )

        assert len(responses) == len(unique_task_ids) == len(co_object_sets), "Number of responses does not match number of tasks in DataFrame"

        for idx, response in enumerate(responses):
            print(f"Response for task {idx}: {response}", flush=True)
            if "error" in response.keys():
                co_object_sets[idx] = {"sets": []}
            else:
                co_object_sets[idx] = response


        # Update the data with the generated object roles
        for i, task in enumerate(unique_task_ids):
            task_rows = df[df['Task ID'] == task]
            for co_object_set in co_object_sets[i].get('sets', []):
                if "score" in task_rows.columns:
                    new_row = task_rows.iloc[0].drop(labels=["score"]).copy()
                else:
                    new_row = task_rows.iloc[0].copy()  # 
                
                new_row["co_object_set"] = json.dumps(co_object_set)
                new_row["Object ID"] = str(uuid.uuid4())
                if system_prompt_name != "image_and_task_instruction":
                    new_row["Attack Image Path"] = f"{task}/{new_row['Object ID']}/final_0.png"
                output.append(new_row)


        output_df = pd.DataFrame(output)

        # Save the updated DataFrame to the specified output path
        output_df.to_csv(output_csv_path, index=False, quoting=1)
        print(f"Updated CSV with co-objects saved to {output_csv_path}.")


    def query_vlm(self, user_prompts, image_paths, system_prompt, temperature, top_p, top_k, seed, max_new_tokens=4096, response_schema=None):
        if self.vlm is None:
            from vlm_2 import VLM  # lazy: needs vllm/torch; the random_list and null_edit modes never get here
            self.vlm = VLM(
                model_name=self.vlm_name,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed
            )
        else:
            self.vlm.set_new_system_prompt(system_prompt)

        responses = self.vlm.query_vlm(
            prompts=user_prompts,
            image_paths=image_paths,
            response_schema=response_schema,
        )

        return responses

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate co-object sets for ER model bias attack")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Path to input CSV file with tasks")
    parser.add_argument("--baseline_evaluation_path", type=str, required=True, help="Path to baseline evaluation CSV file with judgments for filtering")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to output CSV file to save results")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory for images referenced in the CSV")
    parser.add_argument("--vlm_name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8", help="Name of the VLM model to use")
    parser.add_argument("--system_prompt_name", type=str, default="image_and_task_instruction", choices=["image_and_task_instruction", "scene_only", "random_list", "null_edit"], help="Proposer: RQ1 red-team prompt (default), image-only neutral prompt, the fixed random list (no VLM), or empty object sets for the null-edit control (no VLM)")
    parser.add_argument("--n", type=int, default=50, help="Number of co-object sets to generate per task")
    parser.add_argument("--k", type=int, default=1, help="Number of co-objects per object set")
    args = parser.parse_args()

    input_csv_path = args.input_csv_path
    baseline_evaluation_path = args.baseline_evaluation_path
    output_csv_path = args.output_csv_path
    image_root = args.image_root
    system_prompt_name = args.system_prompt_name
    vlm_name = args.vlm_name
    n = args.n
    k = args.k

    attack_stage1 = AttackStage1(vlm_name=vlm_name)
    filtered_df = attack_stage1.filter_from_baseline_evaluation(input_csv_path=input_csv_path, evaluation_csv_path=baseline_evaluation_path)
    attack_stage1.generate_co_objects(df=filtered_df, output_csv_path=output_csv_path, n=n, k=k, image_root=image_root, system_prompt_name=system_prompt_name)