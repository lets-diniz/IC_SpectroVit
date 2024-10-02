import gc
import os
import random

import numpy as np
import torch
from metrics import calculate_metrics
from plot_metrics import PlotMetrics
from tqdm import trange


def valid_on_the_fly(model, epoch, val_dataset, save_dir_path, filename, device):
    model.eval()

    random_index = random.randint(0, len(val_dataset) - 1)
    input_, target, ppm, _, filename = val_dataset[random_index]
    input_ = torch.unsqueeze(input_, dim=0).to(device)

    prediction = model(input_)

    prediction = prediction.cpu().detach().numpy()[0, :]
    target = target.numpy()
    ppm = ppm.numpy()

    os.makedirs(os.path.join(save_dir_path, "spectra_comparison"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_path, "shape_score_comparison"), exist_ok=True)

    PlotMetrics.spectra_comparison(
        prediction,
        target,
        ppm,
        f"{save_dir_path}/spectra_comparison/{filename}_epoch_{epoch + 1}.png",
    )
    PlotMetrics.shape_score_comparison(
        prediction,
        target,
        ppm,
        f"{save_dir_path}/shape_score_comparison/{filename}_epoch_{epoch + 1}.png",
    )


def run_train_epoch(model, optimizer, criterion, loader, epoch, device):
    model.to(device)
    model.train()

    running_loss = 0

    with trange(len(loader), desc="Train Loop") as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            input_, target, ppm = sample_batch[0], sample_batch[1], sample_batch[2]

            input_ = input_.to(device)
            target = target.to(device)
            ppm = ppm.to(device)

            prediction = model(input_)
            loss = criterion(prediction, target, ppm)
            running_loss += loss.item()

            progress_bar.set_postfix(
                desc=(
                    f"[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(loader):d}, "
                    f"loss: {running_loss / (batch_idx + 1)}"
                )
            )

            loss.backward()
            optimizer.step()

        loader_loss = running_loss / len(loader)

    return loader_loss


def run_validation(model, criterion, loader, epoch, device, epsilon=1e-10):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(device)
        model.eval()

        running_loss = 0
        running_mse = 0
        running_snr = 0
        running_linewidth = 0
        running_shape_score = 0

        with trange(len(loader), desc="Validation Loop") as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, loader):
                input_, target, ppm = sample_batch[0], sample_batch[1], sample_batch[2]

                input_ = input_.to(device)
                target = target.to(device)
                ppm = ppm.to(device)

                prediction = model(input_)
                loss = criterion(prediction, target, ppm)

                prediction = prediction.cpu().numpy()
                target = target.cpu().numpy()
                ppm = ppm.cpu().numpy()

                result = calculate_metrics(prediction, target, ppm)

                running_loss += loss.cpu()

                running_mse += np.array(result["mse"]).mean()
                running_snr += np.maximum(0, np.array(result["snr"]).mean())
                running_linewidth += np.array(result["linewidth"]).mean()
                running_shape_score += np.array(result["shape_score"]).mean()

                progress_bar.set_postfix(
                    desc=(
                        f"[Epoch {epoch + 1}] Loss: {running_loss / (batch_idx + 1)} | "
                        f"MSE:{running_mse / (batch_idx + 1):.7f} | "
                        f"SNR:{running_snr / (batch_idx + 1):.7f} | "
                        f"FWHM:{running_linewidth / (batch_idx + 1):.7f} | "
                        f"Shape Score:{running_shape_score / (batch_idx + 1):.7f}"
                    )
                )

    loader_loss = (running_loss / len(loader)).detach().numpy()

    loader_mean_mse = running_mse / len(loader)
    loader_mean_snr = running_snr / len(loader)
    loader_mean_linewidth = running_linewidth / len(loader)
    loader_mean_shape_score = running_shape_score / len(loader)

    score_challenge = 0.8 * loader_mean_mse + 0.2 * (
        1 / (loader_mean_shape_score + epsilon)
    )

    return (
        loader_loss,
        loader_mean_mse,
        loader_mean_snr,
        loader_mean_linewidth,
        loader_mean_shape_score,
        score_challenge,
    )
