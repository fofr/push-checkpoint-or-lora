import tarfile
import json
import io
import os
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


class TrainingOutput(BaseModel):
    weights: Path


def train(
    checkpoint_filename: str = Input(
        description="The filename of the checkpoint to use. "
        "This can be a URL to a tarball containing the weights or a path to the weights file.",
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
    weights_download.check_weight_is_available(checkpoint_filename)

    sampler_node = workflow["3"]["inputs"]
    sampler_node["sampler_name"] = sampler
    sampler_node["steps"] = steps
    sampler_node["cfg"] = cfg
    sampler_node["scheduler"] = scheduler

    checkpoint_loader = workflow["4"]["inputs"]
    checkpoint_loader["ckpt_name"] = checkpoint_filename

    # Create a tar file for the new workflow data
    with tarfile.open("workflow.tar", "w") as tar:
        tarinfo = tarfile.TarInfo(name="workflow_api.json")
        workflow_data = json.dumps(workflow, indent=2).encode("utf-8")
        tarinfo.size = len(workflow_data)
        tar.addfile(tarinfo, io.BytesIO(workflow_data))

    return TrainingOutput(weights=Path("workflow.tar"))
