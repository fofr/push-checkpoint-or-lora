import tarfile
import json
import io
import os
import subprocess
from weights_downloader import WeightsDownloader
from cog import BaseModel, Input, Path
from comfyui_enums import SAMPLERS, SCHEDULERS

os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
api_json_file = "workflow_api.json"


def download_file(url: str, filename: str = "checkpoint.safetensors"):
    if not (
        url.startswith("https://huggingface.co")
        or url.startswith("https://civitai.com")
    ):
        raise ValueError("URL must be from 'huggingface.co' or 'civitai.com'")

    print(f"Downloading {url} to {filename}")
    subprocess.run(["pget", "--log-level", "warn", "-f", url, filename])
    print(f"Successfully downloaded {filename}")
    return filename


class TrainingOutput(BaseModel):
    weights: Path


def train(
    checkpoint: str = Input(
        description="A checkpoint filename that is in https://github.com/fofr/cog-comfyui/blob/main/weights.json. Or a HuggingFace or CivitAI download URL.",
        default="sd_xl_base_1.0.safetensors",
    ),
    lora: str = Input(
        description="Optional: A lora filename that is in https://github.com/fofr/cog-comfyui/blob/main/weights.json. Or a HuggingFace or CivitAI download URL. Optional.",
        default=None,
    ),
    steps: int = Input(
        description="Set the default number of steps to use during inference",
        default=20,
        le=100,
        ge=1,
    ),
    cfg: float = Input(
        description="Set the default CFG scale to use during inference",
        default=7.0,
        le=20.0,
        ge=0.0,
    ),
    sampler: str = Input(
        description="Set the default sampler to use during inference. Choices are: "
        + ", ".join(SAMPLERS),
        default="dpmpp_2m_sde_gpu",
    ),
    scheduler: str = Input(
        description="Set the default scheduler to use during inference. Choices are: "
        + ", ".join(SCHEDULERS),
        default="karras",
    ),
) -> TrainingOutput:
    with open(api_json_file, "r") as file:
        workflow = json.load(file)

    if sampler not in SAMPLERS:
        raise ValueError(f"Sampler {sampler} not found. Must be one of {SAMPLERS}")

    if scheduler not in SCHEDULERS:
        raise ValueError(
            f"Scheduler {scheduler} not found. Must be one of {SCHEDULERS}"
        )

    weights_download = WeightsDownloader()

    sampler_node = workflow["3"]["inputs"]
    sampler_node["sampler_name"] = sampler
    sampler_node["steps"] = steps
    sampler_node["cfg"] = cfg
    sampler_node["scheduler"] = scheduler

    if checkpoint:
        if checkpoint.startswith("https://"):
            checkpoint = download_file(checkpoint, "checkpoint.safetensors")
            checkpoint = "checkpoint.safetensors"
        else:
            weights_download.check_weight_is_available(checkpoint)

        checkpoint_loader = workflow["4"]["inputs"]
        checkpoint_loader["ckpt_name"] = checkpoint

    if lora:
        if lora.startswith("https://"):
            lora = download_file(lora, "lora.safetensors")
            lora = "lora.safetensors"
        else:
            weights_download.check_weight_is_available(lora)

        lora_loader = workflow["10"]["inputs"]
        lora_loader["lora_name"] = lora
    else:
        # Remove lora loading from workflow
        del workflow["10"]

        # Connect sampler to checkpoint loader
        sampler_node["model"] = ["4", 0]

        # Get CLIP from checkpoint loader
        workflow["6"]["inputs"]["clip"] = ["4", 1]
        workflow["7"]["inputs"]["clip"] = ["4", 1]

    # Create a tar file for the new workflow data
    with tarfile.open("weights-and-workflow.tar", "w") as tar:
        tarinfo = tarfile.TarInfo(name="workflow_api.json")
        workflow_data = json.dumps(workflow, indent=2).encode("utf-8")
        tarinfo.size = len(workflow_data)
        tar.addfile(tarinfo, io.BytesIO(workflow_data))

        # Check if checkpoint or lora files are safetensors and add them to the tar if they exist
        for filename in [checkpoint, lora, "updated_weights.json"]:
            if filename and os.path.exists(filename):
                tar.add(filename)

    # Remove any checkpoint or lora files
    for filename in ["checkpoint.safetensors", "lora.safetensors"]:
        if os.path.exists(filename):
            os.remove(filename)

    return TrainingOutput(weights=Path("weights-and-workflow.tar"))
