import tarfile
import json
import io
import os
import subprocess
from weights_downloader import WeightsDownloader
from cog import BaseModel, Input, Path

os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
api_json_file = "workflow_api.json"

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SCHEDULERS = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]


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
    checkpoint_filename: str = Input(
        description="The URL or filename of the checkpoint to use"
    ),
    lora_filename: str = Input(
        description="The URL or filename of the LoRA to use",
    ),
    steps: int = Input(
        description="The number of steps to use during inference",
        default=20,
        le=100,
        ge=1,
    ),
    cfg: float = Input(
        description="The CFG scale to use during inference",
        default=7.0,
        le=20.0,
        ge=0.0,
    ),
    sampler: str = Input(
        description="The sampler to use during inference",
        default="dpmpp_2m_sde_gpu",
    ),
    scheduler: str = Input(
        description="The scheduler to use during inference",
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

    if checkpoint_filename:
        if checkpoint_filename.startswith("https://"):
            checkpoint_filename = download_file(
                checkpoint_filename, "checkpoint.safetensors"
            )
            checkpoint_filename = "checkpoint.safetensors"
        else:
            weights_download.check_weight_is_available(checkpoint_filename)

    if lora_filename:
        if lora_filename.startswith("https://"):
            lora_filename = download_file(lora_filename, "lora.safetensors")
            lora_filename = "lora.safetensors"
        else:
            weights_download.check_weight_is_available(lora_filename)

    sampler_node = workflow["3"]["inputs"]
    sampler_node["sampler_name"] = sampler
    sampler_node["steps"] = steps
    sampler_node["cfg"] = cfg
    sampler_node["scheduler"] = scheduler

    if checkpoint_filename:
        checkpoint_loader = workflow["4"]["inputs"]
        checkpoint_loader["ckpt_name"] = checkpoint_filename

    if lora_filename:
        lora_loader = workflow["10"]["inputs"]
        lora_loader["lora_name"] = lora_filename

    # Create a tar file for the new workflow data
    with tarfile.open("weights-and-workflow.tar", "w") as tar:
        tarinfo = tarfile.TarInfo(name="workflow_api.json")
        workflow_data = json.dumps(workflow, indent=2).encode("utf-8")
        tarinfo.size = len(workflow_data)
        tar.addfile(tarinfo, io.BytesIO(workflow_data))

        # Check if checkpoint or lora files are safetensors and add them to the tar if they exist
        for filename in [checkpoint_filename, lora_filename, "updated_weights.json"]:
            if filename and os.path.exists(filename):
                tar.add(filename)

    # Remove any checkpoint or lora files
    for filename in ["checkpoint.safetensors", "lora.safetensors"]:
        if os.path.exists(filename):
            os.remove(filename)

    return TrainingOutput(weights=Path("weights-and-workflow.tar"))
