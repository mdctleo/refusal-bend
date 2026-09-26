import pandas as pd
from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
from filelock import FileLock
from system_prompts_2 import diffusion_edit_user_prompt
from diffusion_2 import Diffusion


class AttackOriginal():
    def __init__(self):
        self.diffusion = None

    def edit_images(self, attack_csv_path="./attack.csv", rang=None, image_root="./auto", output_path="./auto", num_samples=5):
        df = pd.read_csv(attack_csv_path)

        if rang is not None:
            start, end = rang
            df = df.iloc[start:end]


        image_paths = []
        save_paths = []
        user_prompts = []

        for index, task in enumerate(tqdm(df.to_dict('records'), total=len(df), desc="Preparing image edits")):
            original_image_path = os.path.join(image_root, f"{task['Matched Image Path']}")
            co_objects_set = json.loads(task["co_object_set"])

            for sample_idx in range(num_samples):

                user_prompts.append(diffusion_edit_user_prompt(objects=co_objects_set))
                image_paths.append(original_image_path)

                task_id = task['Task ID']
                object_id = task["Object ID"]
                save_folder = f"{output_path}/{task_id}/{object_id}"
                os.makedirs(save_folder, exist_ok=True)
                save_path = f"{save_folder}/final_{sample_idx}.png"
                save_paths.append(save_path)

        # One image per row -> one relative path per row, written before rendering: a re-render
        # overwrites both the images and their paths.
        if num_samples == 1:
            attack_image_paths = [f"{task['Task ID']}/{task['Object ID']}/final_0.png" for task in df.to_dict('records')]
            self.write_attack_image_paths(attack_csv_path, df.index, attack_image_paths)

        self.edit_images_diffusion(
            prompts=user_prompts,
            image_paths=image_paths,
            save_paths=save_paths,
        )

    def write_attack_image_paths(self, output_path, index, attack_image_paths):
        """Overwrite Attack Image Path for the given rows. Render shards share one CSV, so lock, re-read, write."""
        lock = FileLock(output_path + ".lock")
        with lock:
            df = pd.read_csv(output_path, low_memory=False)
            if "Attack Image Path" not in df.columns:
                df["Attack Image Path"] = ""
            df["Attack Image Path"] = df["Attack Image Path"].astype(object)
            df.loc[index, "Attack Image Path"] = attack_image_paths
            df.to_csv(output_path, index=False)
        print(f"Attack Image Path written for {len(attack_image_paths)} rows to {output_path}", flush=True)

    def edit_images_diffusion(self, prompts, image_paths, save_paths, num_images_per_prompt=1):
        if self.diffusion is None:
            self.diffusion = Diffusion(model_name="flux2-klein", num_inference_steps=4, guidance_scale=1.0, num_images_per_prompt=num_images_per_prompt)

        self.diffusion.edit_images(
            prompts=prompts,
            image_paths=image_paths,
            save_paths=save_paths,
        )

if __name__ == "__main__":
    print("Starting Attack Original Experiment", flush=True)
    parser = ArgumentParser(description="Run attack stage 2 to edit original images based on co-object sets")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Path to input CSV file with attack tasks and co-object sets")
    parser.add_argument("--image_root", type=str, default="./auto", help="Root directory for original images")
    parser.add_argument("--output_path", type=str, default="./attack_images", help="Root directory to save edited images")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of edited images to generate per original image")
    parser.add_argument("--rang", type=str, default=None, help="Range of rows to process in attack stage 2, format: start,end")

    args = parser.parse_args()


    input_csv_path = args.input_csv_path
    image_root = args.image_root
    output_path = args.output_path
    num_samples = args.num_samples
    rang = args.rang
    rang = tuple(map(int, rang.split(","))) if rang is not None else None

    attack_original = AttackOriginal()
    attack_original.edit_images(attack_csv_path=input_csv_path, rang=rang, image_root=image_root, output_path=output_path, num_samples=num_samples)


    
