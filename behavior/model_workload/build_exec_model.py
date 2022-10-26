import math
from plumbum import cli
import argparse
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import joblib
import pickle

import logging
import multiprocessing as mp

import torch
import torch.utils.data.dataset as torch_dataset
from torch.utils.data import DataLoader
import behavior.model_workload.models as models
from behavior.model_workload.models import ModelArgs, MODEL_ARGS_KEYS
from sklearn.model_selection import train_test_split

logger = logging.getLogger("build_exec_model")


def run_job(args):
    # Fix the number of threads possible.
    torch.set_num_threads(int(args.num_threads))

    # Remove the stream handler.
    #logger.getLogger("build_exec_model").handlers.clear()
    (Path(args.output_path)).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(args.output_path) / "output.log", mode="a")
    file_handler.propagate = False
    logger.addHandler(file_handler)

    # Print the arguments.
    logger.info("%s", args)
    MODEL_PATH = f"{args.output_path}/model.pt"
    BEST_MODEL_PATH = f"{args.output_path}/best_model.pt"

    # Get the dataset.
    model_cls = getattr(models, args.model_name)
    if not model_cls.require_optimize():
        # Remove the file handler.
        logger.removeHandler(file_handler)

        # We aren't training a neural network model setup so we use the fit() setup.
        dataset = model_cls.get_dataset(logger, args)
        model_cls(args).fit(dataset)
        open(f"{args.output_path}/outputs.pt", "w").close()
        return

    dataset, feat_names, target_names, num_outputs, classes = model_cls.get_dataset(logger, args)
    target_names.reverse()

    if classes is not None:
        train_size = int(args.train_size * len(dataset[0]))
        val_size = len(dataset[0]) - train_size
        results = train_test_split(*dataset, test_size=int(val_size), train_size=int(train_size), stratify=classes)
        dataset = torch_dataset.TensorDataset(*[torch.tensor(np.array(r, dtype=np.float32)) for r in dataset])

        train_ds = torch_dataset.TensorDataset(*[torch.tensor(np.array(results[r], dtype=np.float32)) for r in range(0, len(results), 2)])
        val_ds = torch_dataset.TensorDataset(*[torch.tensor(np.array(results[r], dtype=np.float32)) for r in range(1, len(results), 2)])
        train_data_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_ds, batch_size=args.batch_size)
        del train_ds
        del val_ds
        del results
    else:
        train_size = int(args.train_size * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        del train_dataset
        del val_dataset

    # Create the model.
    model = model_cls(args, num_outputs)
    if args.cuda:
        model.cuda()

    logger.info("%s", model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epoch = 0
    best_model = None
    best_seen_value = None
    cur_ckpt_loss = None
    last_ckpt_value = None
    if Path(MODEL_PATH).exists():
        ckpt = torch.load(MODEL_PATH)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt["epoch"]
        best_model = ckpt["best_state_dict"]
        best_seen_value = ckpt["best_seen_loss"]
        last_ckpt_value = best_seen_value

    for i in range(epoch, args.num_epochs):
        model.train()

        # Train a mini-batch.
        total_train_loss = 0
        num_batch = math.ceil(train_size / args.batch_size)
        with tqdm(total=num_batch, leave=False) as pbar:
            for batch_idx, data_batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                if args.cuda:
                    data_batch = [f.cuda() for f in data_batch]

                outputs = model(**{k: data_batch[i] for i, k in enumerate(feat_names)})
                loss, it_loss = model.loss({k: data_batch[-(i + 1)] for i, k in enumerate(target_names)}, outputs)
                total_train_loss += it_loss

                loss.backward()
                optimizer.step()
                pbar.update(1)

        # Evaluation.
        total_eval_loss = 0
        num_batch = math.ceil(val_size / args.batch_size)
        with tqdm(total=num_batch, leave=False) as pbar:
            with torch.no_grad():
                for batch_idx, data_batch in enumerate(val_data_loader):
                    if args.cuda:
                        data_batch = [f.cuda() for f in data_batch]

                    outputs = model(**{k: data_batch[i] for i, k in enumerate(feat_names)})
                    _, it_loss = model.loss({k: data_batch[-(i + 1)] for i, k in enumerate(target_names)}, outputs)
                    total_eval_loss += it_loss
                    pbar.update(1)

        logger.info("Epoch %s. Train Loss: %f. Eval Loss %f", i, total_train_loss, total_eval_loss)

        if best_seen_value is None or total_eval_loss < best_seen_value:
            best_seen_value = total_eval_loss
            best_model = copy.deepcopy(model.state_dict())

        if cur_ckpt_loss is None or total_eval_loss < cur_ckpt_loss:
            cur_ckpt_loss = total_eval_loss

        if (i % args.ckpt_interval) == 0 and i > 0:
            torch.save({
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_model,
                "best_seen_loss": best_seen_value,

                "model_args": args,
                "num_outputs": num_outputs,
            }, MODEL_PATH)

            torch.save({
                "model_args": args,
                "num_outputs": num_outputs,
                "best_model": best_model,
            }, BEST_MODEL_PATH)
            logger.info("Best Seen: %s. Last Ckpt Value: %s.", best_seen_value, last_ckpt_value)

        if (i % args.patience) == 0 and i > 0:
            # Break out early in the case that there is no improvement since last ckpt.
            if last_ckpt_value is None or cur_ckpt_loss < last_ckpt_value:
                last_ckpt_value = cur_ckpt_loss
            elif cur_ckpt_loss >= last_ckpt_value:
                break

            cur_ckpt_loss = None

    if best_model is not None:
        torch.save({
            "model_args": args,
            "num_outputs": num_outputs,
            "best_model": best_model,
        }, BEST_MODEL_PATH)

    # Generate final predictions across all train/val data.
    with torch.no_grad():
        self_data_loader = DataLoader(dataset, batch_size=len(dataset))
        for batch_idx, data_batch in enumerate(self_data_loader):
            if args.cuda:
                data_batch = [f.cuda() for f in data_batch]
            outputs = model(**{k: data_batch[i] for i, k in enumerate(feat_names)})

            torch.save(data_batch, f"{args.output_path}/dataset.pt")
            torch.save(outputs, f"{args.output_path}/outputs.pt")

    # Remove the file handler.
    logger.removeHandler(file_handler)


def generate_jobs(model_name, input_dirs, output_dir, lrs, epochs,
                  batch_size, hidden, train_size, cuda, depths,
                  sweep_dropout, add_nonnorm_features, hist_width,
                  num_cpus, max_threads, num_iterations, ckpt_interval,
                  patience, steps, window_slices):
    jobs = []
    for dropout in ([False, True] if sweep_dropout else [False]):
        for epoch in epochs:
            for lr in lrs:
                for hid in hidden:
                    for depth in depths:
                        for batch in batch_size:
                            for it in range(0, num_iterations):
                                l = lr.replace(".", "")
                                s = steps.replace(",", "_")
                                ws = window_slices.replace(",", "_")

                                output = f"{output_dir}/{model_name}_step{s}_window{ws}"
                                if add_nonnorm_features:
                                    output += "_withnonnorm"
                                dataset_path = output

                                output += f"/epochs{epoch}_lr{l}_hid{hid}_batch{batch}_depth{depth}"
                                if dropout:
                                    output += "_dropout"
                                output += f"_{it}"
                                if (Path(output) / "outputs.pt").exists():
                                    continue

                                args = {
                                    "model_name": model_name,
                                    "lr": float(lr),
                                    "num_epochs": int(epoch),
                                    "batch_size": int(batch),
                                    "train_size": train_size,
                                    "cuda": cuda,
                                    "hidden_size": int(hid),
                                    "output_path": output,
                                    "dataset_path": dataset_path,
                                    "depth": int(depth),
                                    "input_dirs": input_dirs,
                                    "dropout": dropout,
                                    "num_threads": min(1, max_threads / num_cpus),
                                    "hist_width": hist_width,
                                    "ckpt_interval": ckpt_interval,
                                    "patience": patience,
                                    "steps": steps.split(","),
                                    "window_slices": window_slices.split(","),
                                    "add_nonnorm_features": add_nonnorm_features,
                                }

                                v = [args[k] for k in MODEL_ARGS_KEYS]
                                args = ModelArgs(*v)
                                jobs.append(args)

    if num_cpus > 1:
        with mp.Pool(num_cpus) as p:
            p.map(run_job, jobs)
    else:
        for j in jobs:
            run_job(j)


class BuildExecModelCLI(cli.Application):
    model_name = cli.SwitchAttr(
        "--model-name",
        str,
        mandatory=True,
        help="Model Name that we should train.",
    )

    lr = cli.SwitchAttr(
        "--lr",
        str,
        mandatory=True,
        help="Learning rate to use for training.",
    )

    epochs = cli.SwitchAttr(
        "--epochs",
        str,
        mandatory=True,
        help="Epochs to use for training.",
    )

    batch_size = cli.SwitchAttr(
        "--batch-size",
        str,
        mandatory=True,
        help="Batch size to use for training.",
    )

    cuda = cli.Flag(
        "--cuda",
        default=False,
        help="Whether to use CUDA or not.",
    )

    train_size = cli.SwitchAttr(
        "--train-size",
        float,
        default=0.8,
        help="Percentage of data to use for training.",
    )

    hidden = cli.SwitchAttr(
        "--hidden",
        str,
        mandatory=True,
        help="Number of hidden units to use across the model.",
    )

    depth = cli.SwitchAttr(
        "--depth",
        str,
        mandatory=True,
        help="Depth to use in the model.",
    )

    input_dirs = cli.SwitchAttr(
        "--input-dirs",
        str,
        mandatory=True,
        help="Path to multiple input directories.",
    )

    output_dir = cli.SwitchAttr(
        "--output-dir",
        Path,
        mandatory=True,
        help="Path to the containing output model directory.",
    )

    add_nonnorm_features = cli.Flag(
        "--add-nonnorm-features",
        default=False,
        help="Whether to add non normalization features.",
    )

    sweep_dropout = cli.Flag(
        "--sweep-dropout",
        default=False,
        help="Whether to sweep dropout.",
    )

    num_iterations = cli.SwitchAttr(
        "--num-iterations",
        int,
        default=1,
        help="Number of iterations.",
    )

    num_cpus = cli.SwitchAttr(
        "--num-cpus",
        int,
        default=mp.cpu_count(),
        help="Number of CPUs to use.",
    )

    max_threads = cli.SwitchAttr(
        "--max-threads",
        int,
        default=mp.cpu_count(),
        help="Maximum number of threads available to allocate.",
    )

    hist_width = cli.SwitchAttr(
        "--hist-width",
        int,
        default=10,
        help="Width of the histogram.",
    )

    ckpt_interval = cli.SwitchAttr(
        "--ckpt-interval",
        int,
        default=100,
        help="Frequency at which checkpointing should happen."
    )

    patience = cli.SwitchAttr(
        "--patience",
        int,
        default=400,
        help="Patience for early stopping."
    )

    steps = cli.SwitchAttr(
        "--steps",
        str,
        default="1",
        help="Default steps to consider for concurrency.",
    )

    window_slices = cli.SwitchAttr(
        "--window-slices",
        str,
        default="1000",
        help="Default slices to consider for buffer page model.",
    )

    def main(self):
        generate_jobs(self.model_name,
                      self.input_dirs.split(","),
                      self.output_dir,
                      self.lr.split(","),
                      self.epochs.split(","),
                      self.batch_size.split(","),
                      self.hidden.split(","),
                      self.train_size,
                      self.cuda,
                      self.depth.split(","),
                      self.sweep_dropout,
                      self.add_nonnorm_features,
                      self.hist_width,
                      self.num_cpus,
                      self.max_threads,
                      self.num_iterations,
                      self.ckpt_interval,
                      self.patience,
                      self.steps,
                      self.window_slices)


if __name__ == "__main__":
    BuildExecModelCLI.run()
