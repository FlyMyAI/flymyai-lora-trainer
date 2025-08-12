from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional
import wandb


class CustomWandbTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: str, project_name: str, entity: Optional[str] = None, tags: Optional[list] = None, config: Optional[dict] = None):
        self.run_name = run_name
        self.project_name = project_name
        self.entity = entity
        self.tags = tags or []
        self.config = config or {}

        # Initialize wandb run
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            entity=self.entity,
            tags=self.tags,
            config=self.config
        )

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        wandb.config.update(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)