import mimetypes
import json
import os
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from safety_checker import SafetyChecker

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"
os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"


class Predictor(BasePredictor):
    def setup(self, weights: str):
        # Weights is a tar containing the workflow
        if not weights:
            print(
                "Warning: Workflow must be provided. "
                "Set COG_WEIGHTS environment variable to "
                "a URL to a tarball containing the workflow file."
            )

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.weights_downloader.download("weights.tar", weights, "")

        if os.path.exists("checkpoint.safetensors"):
            shutil.move(
                "checkpoint.safetensors",
                "ComfyUI/models/checkpoints/checkpoint.safetensors",
            )

        if os.path.exists("lora.safetensors"):
            shutil.move(
                "lora.safetensors", "ComfyUI/models/loras/lora.safetensors"
            )

        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.safetyChecker = SafetyChecker()

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(workflow)

    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        empty_latent_image = workflow["5"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["number_of_images"]

        lora_loader = workflow["10"]["inputs"]
        lora_loader["strength_model"] = kwargs["lora_strength"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        number_of_images: int = Input(
            description="Number of images to generate", ge=1, le=10, default=1
        ),
        width: int = Input(default=1024),
        height: int = Input(default=1024),
        lora_strength: float = Input(default=1.0, ge=0, le=3.0),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.", default=False
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        using_fixed_seed = bool(seed)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            number_of_images=number_of_images,
            lora_strength=lora_strength,
        )

        self.comfyUI.connect()

        try:
            if using_fixed_seed:
                self.comfyUI.reset_execution_cache()
        except Exception as e:
            print(f"Failed to reset execution cache: {e}")

        self.comfyUI.run_workflow(workflow)

        files = self.comfyUI.get_files(OUTPUT_DIR)

        if not disable_safety_checker:
            has_nsfw_content = self.safetyChecker.run(files)
            if any(has_nsfw_content):
                print("Removing NSFW images")
                files = [f for i, f in enumerate(files) if not has_nsfw_content[i]]

        return optimise_images.optimise_image_files(
            output_format, output_quality, files
        )
