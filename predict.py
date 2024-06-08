import mimetypes
import json
import os
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from comfyui_enums import SAMPLERS, SCHEDULERS
from safety_checker import SafetyChecker

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

api_json_file = "workflow_api.json"


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
            os.makedirs("ComfyUI/models/checkpoints", exist_ok=True)
            shutil.move(
                "checkpoint.safetensors",
                "ComfyUI/models/checkpoints/checkpoint.safetensors",
            )

        if os.path.exists("lora.safetensors"):
            os.makedirs("ComfyUI/models/loras", exist_ok=True)
            shutil.move("lora.safetensors", "ComfyUI/models/loras/lora.safetensors")

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

        if kwargs["sampler"] != "Default":
            sampler["sampler_name"] = kwargs["sampler"]

        if kwargs["scheduler"] != "Default":
            sampler["scheduler"] = kwargs["scheduler"]

        if kwargs["steps"] is not None:
            sampler["steps"] = kwargs["steps"]

        if kwargs["cfg"] is not None:
            sampler["cfg"] = kwargs["cfg"]

        empty_latent_image = workflow["5"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["number_of_images"]

        if "10" in workflow:
            lora_loader = workflow["10"]["inputs"]
            lora_loader["strength_model"] = kwargs["lora_strength"]

        print("==============================")
        print("Generation settings")
        print("Sampler:", sampler["sampler_name"])
        print("Scheduler:", sampler["scheduler"])
        print("Steps:", sampler["steps"])
        print("CFG:", sampler["cfg"])
        if "10" in workflow:
            print("LORA: Using a lora. Lora strength:", lora_loader["strength_model"])
        else:
            print("LORA: No lora. Lora strength has no effect")
        print("==============================")

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
        lora_strength: float = Input(
            default=1.0,
            ge=0,
            le=3.0,
            description="Strength of the lora to use for the generation. Default is 1.0.",
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.", default=False
        ),
        sampler: str = Input(
            default="Default",
            choices=["Default"] + SAMPLERS,
            description="Advanced. Change the sampler used for generation. Default is what we think gives the best images.",
        ),
        scheduler: str = Input(
            default="Default",
            choices=["Default"] + SCHEDULERS,
            description="Advanced. Change the scheduler used for generation. Default is what we think gives the best images.",
        ),
        steps: int = Input(
            default=None,
            ge=1,
            le=50,
            description="Advanced. Leave empty to use recommended steps. Set it only if you want to customise the number of steps to run the sampler for.",
        ),
        cfg: float = Input(
            default=None,
            ge=0,
            le=20,
            description="Advanced. Leave empty to use recommended CFG (classifier free guidance). This changes how much the prompt influences the output. Set it only if you want to customise.",
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
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
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
