from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "gpt-neo"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
    "n_y": merge(tinteger, default(1)),
    "model_name": merge(
        tstring,
        allowed(
            [
                "EleutherAI/gpt-neo-1.3B",
                "EleutherAI/gpt-neo-125m",
                "EleutherAI/pythia-1.4b-deduped",
                "EleutherAI/pythia-2.8b-deduped",
                "facebook/opt-iml-max-1.3b",
            ]
        ),
        nullable,
    ),
    "lr_solver_head": merge(tboolean, default(False)),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
    "probabilities": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "probabilistic_logistic_regression",
    "nladap",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian", "multigaussian", "nl", "nlreal"])),
    "batch_size": merge(tinteger, default(64)),
    "weight_multiplier": merge(tinteger, default(1)),
    "variable_noise": merge(tboolean, default(False)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "lr_scheduler": merge(tinteger, default(500)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "per_device_batch_size": merge(tinteger, default(1)),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
