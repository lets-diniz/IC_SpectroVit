"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import argparse
import gc
import os
import random
import numpy as np
import shutil
import wandb
from tqdm import trange
from constants import *
from lr_scheduler import CustomLRScheduler
from metrics import calculate_metrics
from utils import read_yaml
from plot_metrics import PlotMetrics

plot_metrics = PlotMetrics()


def valid_on_the_fly(model, epoch, configs, save_dir_path):
    model.eval()

    val_dataset_configs = configs["valid_dataset"]
    val_dataset = FACTORY_DICT["dataset"][list(val_dataset_configs)[0]](
        **val_dataset_configs[list(val_dataset_configs.keys())[0]])

    random_index = random.randint(0, len(val_dataset) - 1)
    input, target, ppm, filename = val_dataset[random_index]
    input = torch.unsqueeze(input, dim=0).to(DEVICE)

    prediction = model(input)

    prediction = prediction.cpu().detach().numpy()[0, :]
    target = target.numpy()
    ppm = ppm.numpy()

    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(os.path.join(save_dir_path, "spectra_comparison"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_path, "shape_score_comparison"), exist_ok=True)

    filename = filename.split(".")[0]
    PlotMetrics.spectra_comparison(prediction, target, ppm,
                                   f"{save_dir_path}/spectra_comparison/{filename}_epoch_{epoch+1}.png")
    PlotMetrics.shape_score_comparison(prediction, target, ppm,
                                       f"{save_dir_path}/shape_score_comparison/{filename}_{epoch + 1}.png")


class ToolsWandb:
    @staticmethod
    def config_flatten(config, parent_key='', sep='_'):
        items = []
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(ToolsWandb.config_flatten(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def get_dataset(dataset_configs):
    dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def set_samples_dataset(configs, samples, type_dataset='train_dataset', key_data="path_data"):
    configs[type_dataset][list(configs[type_dataset].keys())[0]][key_data] = samples
    return configs


def set_length_dataset(configs, len_, type_dataset='train_dataset', key_data="length_dataset"):
    configs[type_dataset][list(configs[type_dataset].keys())[0]][key_data] = len_
    return configs


def experiment_factory(configs):
    train_dataset_configs = configs["train_dataset"]
    train_dataset_key = list(configs["train_dataset"].keys())[0]

    validation_dataset_configs = configs["valid_dataset"]
    validation_dataset_key = list(configs["valid_dataset"].keys())[0]

    if (not isinstance(train_dataset_configs[train_dataset_key]["path_data"], list)) and (
            not isinstance(validation_dataset_configs[validation_dataset_key]["path_data"], list)):
        print(f"length train: {len(os.listdir(train_dataset_configs[train_dataset_key]['path_data']))}")
        print(f"length validation: {len(os.listdir(validation_dataset_configs[validation_dataset_key]['path_data']))}")

    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    train_dataset = get_dataset(train_dataset_configs)
    validation_dataset = get_dataset(validation_dataset_configs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs["train"]["batch_size"], shuffle=True,
        num_workers=configs["train"]["num_workers"]
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=configs["valid"]["batch_size"], shuffle=False,
        num_workers=configs["valid"]["num_workers"]
    )

    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
        model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
    )

    if type(criterion_configs) == dict:
        criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]](
            **criterion_configs[list(criterion_configs.keys())[0]]
        )
    else:
        criterion = FACTORY_DICT["loss"][criterion_configs]()

    return model, train_loader, validation_loader, optimizer, \
        criterion


def run_train_epoch(model, optimizer, criterion, loader,
                    epoch):
    model.to(DEVICE)
    model.train()

    running_loss = 0

    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            input, target, ppm = sample_batch[0], sample_batch[1], sample_batch[2]

            input = input.to(DEVICE)
            target = target.to(DEVICE)
            ppm = ppm.to(DEVICE)

            prediction = model(input)
            loss = criterion(prediction, target, ppm)
            running_loss += loss.item()

            progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, '
                      f'loss: {running_loss / (batch_idx + 1)}')
            )

            loss.backward()
            optimizer.step()

            if configs['wandb']["activate"]:
                wandb.log({'train_loss': loss})

        running_loss = (running_loss / len(loader))

    return running_loss


def run_validation(model, criterion, loader,
                   epoch, configs, epsilon=1e-5):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(DEVICE)
        model.eval()

        running_loss = 0
        running_mse = 0
        running_snr = 0
        running_linewidth = 0
        running_shape_score = 0

        with trange(len(loader), desc='Validation Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, loader):
                input, target, ppm = sample_batch[0], sample_batch[1], sample_batch[2]

                input = input.to(DEVICE)
                target = target.to(DEVICE)
                ppm = ppm.to(DEVICE)

                prediction = model(input)
                loss = criterion(prediction, target, ppm)

                prediction = prediction.cpu().numpy()
                target = target.cpu().numpy()
                ppm = ppm.cpu().numpy()

                result = calculate_metrics(prediction, target, ppm)

                running_loss += loss.cpu()

                running_mse += np.array(result['mse']).mean()
                running_snr += np.maximum(0, np.array(result['snr']).mean())
                running_linewidth += np.array(result['linewidth']).mean()
                running_shape_score += np.array(result['shape_score']).mean()

                progress_bar.set_postfix(

                    desc=(f"[Epoch {epoch + 1}] Loss: {running_loss / (batch_idx + 1)} | "
                          f"MSE:{running_mse / (batch_idx + 1):.7f} | "
                          f"SNR:{running_snr / (batch_idx + 1):.7f} | "
                          f"FWHM:{running_linewidth / (batch_idx + 1):.7f} | "
                          f"Shape Score:{running_shape_score / (batch_idx + 1):.7f}")
                )

    loader_loss = (running_loss / len(loader)).detach().numpy()

    loader_mean_mse = running_mse / len(loader)
    loader_mean_snr = running_snr / len(loader)
    loader_mean_linewidth = running_linewidth / len(loader)
    loader_mean_shape_score = running_shape_score / len(loader)

    score_challenge = (0.8 * loader_mean_mse + 0.2 * (1 / loader_mean_shape_score + epsilon))

    if configs['wandb']["activate"]:
        wandb.log({'mean_valid_loss': loss})
        wandb.log({'mean_mse': loader_mean_mse})
        wandb.log({'mean_snr': loader_mean_snr})
        wandb.log({'mean_linewidth': loader_mean_linewidth})
        wandb.log({'mean_shape_score': loader_mean_shape_score})
        wandb.log({"score_challenge": score_challenge})

    if configs['current_model']['save_model']:
        save_path_model = f"{configs['current_model']['model_dir']}/{configs['current_model']['model_name']}.pt"
        save_best_model(score_challenge, model, save_path_model)

    if configs["valid_on_the_fly"]["activate"]:
        valid_on_the_fly(model, epoch, configs, configs["valid_on_the_fly"]["save_dir_path"])
    return loader_loss


def get_params_lr_scheduler(configs):
    activate = bool(configs["lr_scheduler"]["activate"])
    scheduler_kwargs = configs["lr_scheduler"]["info"]
    scheduler_type = configs["lr_scheduler"]["scheduler_type"]
    return activate, scheduler_type, scheduler_kwargs


def calculate_parameters(model):
    qtd_model = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {qtd_model}")
    return


def run_training_experiment(model, train_loader, validation_loader, optimizer, custom_lr_scheduler,
                            criterion, configs
                            ):
    if configs['current_model']['save_model']:
        os.makedirs(configs['current_model']['model_dir'], exist_ok=True)

    calculate_parameters(model)

    for epoch in range(0, configs["epochs"]):
        train_loss = run_train_epoch(
            model, optimizer, criterion, train_loader,
            epoch
        )

        valid_loss = run_validation(
            model, criterion, validation_loader,
            epoch, configs
        )
        if custom_lr_scheduler is not None:
            if custom_lr_scheduler.scheduler_type == "reducelronplateau":
                custom_lr_scheduler.step(valid_loss)
            else:
                custom_lr_scheduler.step()
            print("Current learning rate:", custom_lr_scheduler.scheduler.get_last_lr()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    print("============ Delete .wandb path ============")
    try:
        shutil.rmtree("wandb/")
    except:
        pass

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs)

    model, train_loader, validation_loader, \
        optimizer, criterion = experiment_factory(configs)

    activate_lr_scheduler, scheduler_type, scheduler_kwargs = get_params_lr_scheduler(configs)

    if activate_lr_scheduler:
        custom_lr_scheduler = CustomLRScheduler(optimizer, scheduler_type, **scheduler_kwargs)
    else:
        custom_lr_scheduler = None

    if configs['reload_from_existing_model']['activate']:
        name_model = f"{configs['reload_from_existing_model']['model_dir']}/{configs['reload_from_existing_model']['model_name']}.pt"

        load_dict = torch.load(name_model)

        model.load_state_dict(load_dict['model_state_dict'])

    if configs['wandb']["activate"]:
        wandb.init(project=configs['wandb']["project"],
                   reinit=True,
                   config=f_configurations,
                   entity=configs['wandb']["entity"],
                   save_code=False)

    run_training_experiment(
        model, train_loader, validation_loader, optimizer, custom_lr_scheduler,
        criterion, configs
    )

    torch.cuda.empty_cache()
    wandb.finish()
