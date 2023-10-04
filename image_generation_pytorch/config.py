from typing import Literal


class Config:
    num_epochs: int
    batch_size: int
    sample_size: int
    num_workers: int
    lr: float
    beta1: float
    beta2: float
    g_layers: list[int]
    d_layers: list[int]
    activation_fn: bool
    inject_z: bool
    concat_injection: bool
    loss: Literal['original', 'wgan-gp']
    gp_weight: float
    seed: int
    pc_name: str
    spectral_norm: bool
    transform_z: bool
    transform_rep: int
    norm: Literal['batch', 'instance']
    mode: Literal['train', 'test', 'sample']
    model_path: str
    sample_path: str
    logs_path: str
    data_path: str
    base_path: str
    ckpt_gen_path: str
    log_step: int
    sample_step: int
    validation_step: int
    validation_path: str
    cifar10_path: str
    FID_images: int
    num_imgs_val: int
    max_score: float
    save_every: int
    n_samples: int
