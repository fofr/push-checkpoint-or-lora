import mimetypes
import json
import os
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper
from comfyui_enums import (
    SAMPLERS,
    SCHEDULERS,
    CONTROLNET_PREPROCESSORS,
    CONTROLNET_MODELS,
)
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

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        is_img2img = kwargs["image_filename"] is not None
        is_controlnet = kwargs["controlnet_image_filename"] is not None

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

        if is_img2img:
            load_image = workflow["13"]["inputs"]
            load_image["image"] = kwargs["image_filename"]
            image_resize = workflow["12"]["inputs"]
            image_resize["width"] = kwargs["width"]
            image_resize["height"] = kwargs["height"]
            repeat_latent_batch = workflow["16"]["inputs"]
            repeat_latent_batch["amount"] = kwargs["number_of_images"]
            sampler["denoise"] = kwargs["denoise_strength"]

        else:
            sampler["latent_image"] = ["5", 0]
            sampler["denoise"] = 1

            empty_latent_image = workflow["5"]["inputs"]
            empty_latent_image["width"] = kwargs["width"]
            empty_latent_image["height"] = kwargs["height"]
            empty_latent_image["batch_size"] = kwargs["number_of_images"]

            """
            Delete:
            - load image
            - image resize
            - vae encode
            - latent repeat batch
            """
            del workflow["12"]
            del workflow["13"]
            del workflow["14"]
            del workflow["16"]

        if is_controlnet:
            preprocessor = workflow["17"]["inputs"]
            preprocessor["preprocessor"] = kwargs["controlnet_preprocessor"]
            load_controlnet_image = workflow["18"]["inputs"]
            load_controlnet_image["image"] = kwargs["controlnet_image_filename"]
            controlnet = workflow["19"]["inputs"]
            controlnet["control_net_name"] = kwargs["controlnet_model"]
            apply_controlnet = workflow["22"]["inputs"]
            apply_controlnet["strength"] = kwargs["controlnet_strength"]
            apply_controlnet["start_percent"] = kwargs["controlnet_start"]
            apply_controlnet["end_percent"] = kwargs["controlnet_end"]

            if not kwargs["controlnet_return_preprocessor_image"]:
                del workflow["24"]

        else:
            sampler["positive"] = ["6", 0]
            sampler["negative"] = ["7", 0]

            """
            Delete:
            - load image
            - preprocessor
            - load controlnet
            - apply controlnet
            - save controlnet
            """
            del workflow["17"]
            del workflow["18"]
            del workflow["19"]
            del workflow["22"]
            del workflow["24"]

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

        if is_controlnet:
            print("Using controlnet")
            print("Controlnet: Model: ", controlnet["control_net_name"])
            print("Controlnet: Preprocessor: ", preprocessor["preprocessor"])
            print("Controlnet: Strength: ", apply_controlnet["strength"])
            print("Controlnet: Start: ", apply_controlnet["start_percent"])
            print("Controlnet: End: ", apply_controlnet["end_percent"])
        if is_img2img:
            print("Using image2image")
            print("Image2Image: Max width:", image_resize["width"])
            print("Image2Image: Max height:", image_resize["height"])
            print("Image2Image: Denoise strength:", sampler["denoise"])
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
        image: Path = Input(
            default=None,
            description="Optional. Image to use for img2img generation. Leave empty to use just a prompt.",
        ),
        denoise_strength: float = Input(
            default=0.65,
            ge=0,
            le=1.0,
            description="How much of the original input image to destroy when using img2img. 1 is total destruction. 0 is untouched. 0.65 is a good balance.",
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
        controlnet_image: Path = Input(
            default=None,
            description="Advanced. The image to use with a controlnet. Leave empty to use no controlnet.",
        ),
        controlnet_preprocessor: str = Input(
            default="None",
            choices=["None"] + CONTROLNET_PREPROCESSORS,
            description="Advanced. Change the preprocessor used for the controlnet. Make sure it matches the controlnet model you're using.",
        ),
        controlnet_return_preprocessor_image: bool = Input(
            default=False,
            description="Advanced. Return the processed image from the controlnet preprocessor. Useful for debugging.",
        ),
        controlnet_model: str = Input(
            default="None",
            choices=["None"] + CONTROLNET_MODELS,
            description="Advanced. Change the model used for the controlnet. Make sure it matches the checkpoint type (eg SD15 or SDXL) and pick an appropriate controlnet preprocessor to go with it.",
        ),
        controlnet_strength: float = Input(
            default=1.0,
            ge=0,
            le=3.0,
            description="Advanced. Strength of the controlnet. Default is 1.0.",
        ),
        controlnet_start: float = Input(
            default=0.0,
            ge=0,
            le=1.0,
            description="Advanced. When to begin applying the controlnet.",
        ),
        controlnet_end: float = Input(
            default=1.0,
            ge=0,
            le=1.0,
            description="Advanced. When to stop applying the controlnet.",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        using_fixed_seed = bool(seed)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        if image:
            file_extension = os.path.splitext(image)[1].lower()
            image_filename = f"input{file_extension}"
            self.handle_input_file(image, image_filename)
        else:
            image_filename = None

        if controlnet_image:
            file_extension = os.path.splitext(controlnet_image)[1].lower()
            controlnet_image_filename = f"controlnet{file_extension}"
            self.handle_input_file(controlnet_image, controlnet_image_filename)
        else:
            controlnet_image_filename = None

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            number_of_images=number_of_images,
            lora_strength=lora_strength,
            image_filename=image_filename,
            denoise_strength=denoise_strength,
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            controlnet_image_filename=controlnet_image_filename,
            controlnet_preprocessor=controlnet_preprocessor,
            controlnet_model=controlnet_model,
            controlnet_strength=controlnet_strength,
            controlnet_start=controlnet_start,
            controlnet_end=controlnet_end,
            controlnet_return_preprocessor_image=controlnet_return_preprocessor_image,
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
