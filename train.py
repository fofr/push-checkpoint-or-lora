import tarfile
import json
from cog import BaseModel, Input, Path

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
    weights_string: str = Input(
        description="A string containing the weights to download. "
        "This can be a URL to a tarball containing the weights or a path to the weights file.",
    ),
    steps: int = Input(
        description="The number of steps to use during inference", default=20
    ),
    cfg: float = Input(
        description="The CFG scale to use during inference", default=7.0
    ),
    sampler: str = Input(
        description="The sampler to use during inference",
        choices=SAMPLERS,
        default="dpmpp_2m_sde_gpu",
    ),
    scheduler: str = Input(
        description="The scheduler to use during inference",
        choices=SCHEDULERS,
        default="karras",
    ),
) -> TrainingOutput:
    with open(api_json_file, "r") as file:
        workflow = json.load(file)

    sampler_node = workflow["3"]["inputs"]
    sampler_node["sampler_name"] = sampler
    sampler_node["steps"] = steps
    sampler_node["cfg"] = cfg
    sampler_node["scheduler"] = scheduler

    checkpoint_loader = workflow["4"]["inputs"]
    checkpoint_loader["ckpt_name"] = weights_string

    # Write the modified data to a new JSON file
    with open("new_workflow_api.json", "w") as new_file:
        json.dump(workflow, new_file, indent=4)

    # Create a tar file for the new workflow data
    with tarfile.open("workflow.tar", "w") as tar:
        tar.add("new_workflow_api.json", arcname="new_workflow_data.json")

    return TrainingOutput(weights=Path("workflow.tar"))
