import dataclasses
import configparser
import os

MODEL_SECTION = 'Model'
TRAINING_SECTION = 'Training'


@dataclasses.dataclass
class Config:
    scale: int = None
    classifier_regressor_kernel_size: int = None
    delta: int = None
    channel_attention_module: bool = None
    fpn_lateral_depth: int = None
    classifier_regressor_depth: int = None
    max_detections: int = None
    nms_threshold: float = None
    player_threshold: float = None
    path_to_model: str = None

    dataset_root_dir: str = None
    dataset_quality: str = None
    train_batch: int = None
    test_batch: int = None
    dataloader_workers: int = None
    model_name: str = None

    learning_rate: float = None
    epochs: int = None
    lr_scheduler_milestones: list = None
    output_dir: str = None

    @classmethod
    def from_configfile(cls, config_path, training=True):
        assert os.path.exists(config_path), "Passed config path must exist"
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        model_config = config_parser[MODEL_SECTION]
        if training:
            model_config.update(config_parser[TRAINING_SECTION])
        return cls(**model_config)

    def __post_init__(self):
        fields = dataclasses.fields(self)
        for field in fields:
            current_value = self.__getattribute__(field.name)
            if current_value is None:
                continue
            new_value = field.type(current_value)
            if field.name == "channel_attention_module":
                assert current_value in ["yes", "no"], "channel_attention_module can only be 'yes' or 'no'"
                new_value = current_value == "yes"
            elif field.name == "lr_scheduler_milestones":
                values = current_value.replace("[", "").replace("]", "").split(",")
                new_value = [int(value) for value in values]

            self.__setattr__(field.name, new_value)

