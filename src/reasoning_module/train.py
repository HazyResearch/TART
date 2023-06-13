import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml


from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(
    model, xs, ys, optimizer, loss_func, batch_size, per_device_batch_size, i_accumulate
):
    acc_steps = batch_size / (per_device_batch_size)
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss = loss / (acc_steps)
    loss.backward()
    if (i_accumulate + 1) % (acc_steps) == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    per_device_batch_size = args.training.per_device_batch_size
    batch_size = args.training.batch_size

    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        batch_size,
        num_tasks=args.training.num_tasks,
        weight_multiplier=args.training.weight_multiplier,
        variable_noise=args.training.variable_noise,
        n_points=args.model.n_positions,
        tokenizer_name=args.model.model_name,
        **args.training.task_kwargs,
    )
    accumulation_steps = batch_size / per_device_batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    optimizer.zero_grad()
    for i in pbar:
        loss_total = 0
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= batch_size
            seeds = sample_seeds(num_training_examples, batch_size)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs, theta = data_sampler.sample_xs(
            curriculum.n_points,
            batch_size,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)

        if args.training.task in [
            "probabilistic_logistic_regression",
        ]:
            ys, wb = task.evaluate(xs, theta)
        elif args.training.task in ["nl", "nlreal", "nladap"]:
            ys, wbs, nl_str = task.evaluate(xs, theta)
            xs = nl_str

        loss_func = task.get_training_metric()

        for i_accumulate in range(int(accumulation_steps)):
            xs_sample = xs[
                i_accumulate
                * per_device_batch_size : (i_accumulate + 1)
                * per_device_batch_size,
                :,
            ]
            ys_sample = ys[
                i_accumulate
                * per_device_batch_size : (i_accumulate + 1)
                * per_device_batch_size,
                :,
            ]

            loss, output = train_step(
                model,
                xs_sample.cuda(),
                ys_sample.cuda(),
                optimizer,
                loss_func,
                batch_size,
                per_device_batch_size,
                i_accumulate,
            )
            loss_total += loss

        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += torch.norm(p.grad.data)
        grad_norm = grad_norm

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "grad_norm": grad_norm,
                    "overall_loss": loss_total,
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss_total}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "gpt-neo"]
    print(f"Running with: {args}")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
