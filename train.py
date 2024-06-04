import tarfile
import json
import io
from cog import BaseModel, Input, Path

api_json_file = "workflow_api.json"


class TrainingOutput(BaseModel):
    weights: Path


def train(
    checkpoint_filename: str = Input(
        description="The filename of the checkpoint to use. "
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
        default="dpmpp_2m_sde_gpu",
    ),
    scheduler: str = Input(
        description="The scheduler to use during inference",
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
    checkpoint_loader["ckpt_name"] = checkpoint_filename

    # Create a tar file for the new workflow data
    with tarfile.open("workflow.tar", "w") as tar:
        tarinfo = tarfile.TarInfo(name="workflow_api.json")
        workflow_data = json.dumps(workflow, indent=2).encode("utf-8")
        tarinfo.size = len(workflow_data)
        tar.addfile(tarinfo, io.BytesIO(workflow_data))

    return TrainingOutput(weights=Path("workflow.tar"))
