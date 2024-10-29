import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import random
random.seed(5)
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal,stats
import os
import csv
import wandb

from models import SpectroViT
from lr_scheduler import CustomLRScheduler
from main_functions_adapted import valid_on_the_fly, run_train_epoch, run_validation
from main import calculate_parameters
from utils import clean_directory, read_yaml, retrieve_metrics_from_csv, plot_training_evolution
from save_models import safe_save, delete_safe_save, save_trained_model
from constants import *


config_path = input("Enter path for config.yaml: ")
config = read_yaml(file=config_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####----------------------Definition-----------------------------------
#names and directories
name_model = str(config['model']['name_model'])
save_dir_path = str(config['model'].get('dir_save_results', f'./{name_model}/'))
dir_save_models = save_dir_path+'models/'
new_model = bool(config['model'].get('new_model', True))

#data
data_folder = str(config['data'].get('data_folder','../sample_data.h5'))
#train data
start_point_train_data = int(config['data']['start_point_train_data'])
end_point_train_data = int(config['data']['end_point_train_data'])
augment_with_noise_train = bool(config['data']['augment_with_noise_train'])
augment_with_idx_repetition_train = bool(config['data']['augment_with_idx_repetition_train'])
qntty_to_augment_by_idx_train = config['data'].get('qntty_to_augment_by_idx_train',None)
#validation data
start_point_validation_data = int(config['data']['start_point_validation_data'])
end_point_validation_data = int(config['data']['end_point_validation_data'])
augment_with_noise_validation = bool(config['data']['augment_with_noise_validation'])
augment_with_idx_repetition_validation = bool(config['data']['augment_with_idx_repetition_validation'])
qntty_to_augment_by_idx_validation = config['data'].get('qntty_to_augment_by_idx_validation',None)
augment_by_noise_kwargs = config['data'].get('random_augment',{})

#spgram characteristics
hop_size = config['data']['hop_size']
window_size = FACTORY_DICT["spgram_window"]["window_size"]
window = FACTORY_DICT["spgram_window"]["window"]

#training
batch_size_train = int(config['training']['batch_size_train'])
batch_size_validation = int(config['training']['batch_size_validation'])
n_epochs = int(config['training']['n_epochs'])

#loss
criterion_configs = config["loss"]
if type(criterion_configs) == dict:
  loss = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]](
            **criterion_configs[list(criterion_configs.keys())[0]]
        )
else:
  loss = FACTORY_DICT["loss"][criterion_configs]()

#model
spectrovit = SpectroViT().to(device)

#optimizer
optimizer_configs = config['optimizer']
initial_lr = optimizer_configs[list(optimizer_configs.keys())[0]]['lr']
optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
              spectrovit.parameters(), 
              **optimizer_configs[list(optimizer_configs.keys())[0]])

#lr scheduler
use_lr_scheduler = bool(config['lr_scheduler']['activate'])
if use_lr_scheduler is True:
    scheduler_type = config['lr_scheduler']['scheduler_type']
    epoch_to_switch_to_lr_scheduler = int(config['lr_scheduler']['epoch_to_switch_to_lr_scheduler'])
    initial_lr_scheduler = float(config['lr_scheduler']['initial_lr_scheduler'])
    model_scheduler = CustomLRScheduler(optimizer,scheduler_type,**config['lr_scheduler'].get('info',{}))
else:
    scheduler_type = None
    model_scheduler = None
    epoch_to_switch_to_lr_scheduler = None
    initial_lr_scheduler = None

#saves
step_to_safe_save_models = int(config['save_models_and_results']['step_to_safe_save_models'])
step_for_saving_plots = int(config['save_models_and_results']['step_for_saving_plots'])
save_losses_and_metrics = FACTORY_DICT["savelossesandmetrics"]["SaveLossesAndMetrics"](dir_save_results=save_dir_path)
save_best_model_bool = bool(config['save_models_and_results']['save_best_model'])
if save_best_model_bool is True:
    best_model = FACTORY_DICT["savebest"]["SaveBestModelState"](dir_save_model=dir_save_models)

#wandb
use_wandb = bool(config['wandb']['activate'])
if use_wandb is True:
  wandb.init(
        # set the wandb project where this run will be logged
        project=name_model,
        # track hyperparameters and run metadata
        config={
            "datafolder": data_folder,
            "idx_initial_train_data": start_point_train_data,
            "idx_final_train_data": end_point_train_data,
            "idx_initial_val_data": start_point_validation_data,
            "idx_final_val_data": end_point_validation_data,
            "batch_size_train": batch_size_train,
            "batch_size_val": batch_size_validation,
            "epochs": n_epochs,
            "criterion": config['loss'],
            "optimizer": config['optimizer'],
            "initial_lr": initial_lr,
            "scheduler": scheduler_type,
            "epoch_to_switch_to_lr_scheduler": epoch_to_switch_to_lr_scheduler
        }
    )


####----------------------Preparing objects-----------------------------------
dataset_train = FACTORY_DICT["dataset"]["DatasetSpgramSyntheticData"](path_data=data_folder,
                                                                      augment_with_noise=augment_with_noise_train,
                                                                      augment_with_idx_repetition=augment_with_idx_repetition_train,
                                                                      start=start_point_train_data,
                                                                      end=end_point_train_data,
                                                                      hop_size=hop_size,
                                                                      window_size=window_size,
                                                                      window=window,
                                                                      qntty_to_augment_by_idx=qntty_to_augment_by_idx_train,
                                                                      **augment_by_noise_kwargs)

dataset_validation = FACTORY_DICT["dataset"]["DatasetSpgramSyntheticData"](path_data=data_folder,
                                                                      augment_with_noise=augment_with_noise_validation,
                                                                      augment_with_idx_repetition=augment_with_idx_repetition_validation,
                                                                      start=start_point_validation_data,
                                                                      end=end_point_validation_data,
                                                                      hop_size=hop_size,
                                                                      window_size=window_size,
                                                                      window=window,
                                                                      qntty_to_augment_by_idx=qntty_to_augment_by_idx_validation,
                                                                      **augment_by_noise_kwargs)


dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=True)

train_loss_list = []
val_loss_list = []
val_mean_mse_list = []
val_mean_snr_list = []
val_mean_linewidth_list = []
val_mean_shape_score_list = []
score_challenge_list = []

os.makedirs(save_dir_path, exist_ok=True)
if new_model is True:
  clean_directory(save_dir_path)
os.makedirs(dir_save_models, exist_ok=True)
if new_model is True:
    with open(save_dir_path+'losses_and_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['LossTrain', 'LossVal', 'MSEVal', 'SNRVal','FWHMVal','ShScVal','ChalScVal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
else:
    print('Loading old model to keep training...')
    if 'path_to_saved_model' in config['model']:
      if config['model']['path_to_saved_model'] != "":
        spectrovit.load_state_dict(torch.load(str(config['model']['path_to_saved_model']), weights_only=True))
      else:
        spectrovit.load_state_dict(torch.load(dir_save_models+f"{name_model}_savesafe.pt", weights_only=True))
    else:
        spectrovit.load_state_dict(torch.load(dir_save_models+f"{name_model}_savesafe.pt", weights_only=True))
    if 'path_to_saved_optimizer' in config['optimizer']:
      if config['optimizer']['path_to_saved_optimizer'] != "":
        optimizer.load_state_dict(torch.load(str(config['optimizer']['path_to_saved_optimizer'],weights_only=True)))
      else:
        optimizer.load_state_dict(torch.load(dir_save_models+f"{name_model}_optimizer_savesafe.pt",weights_only=True))
    else:
        optimizer.load_state_dict(torch.load(dir_save_models+f"{name_model}_optimizer_savesafe.pt",weights_only=True))
    if use_lr_scheduler is True:
      if 'path_to_saved_scheduler' in config['lr_scheduler']:
        if config['lr_scheduler']['path_to_saved_scheduler'] != "":
          model_scheduler.load_state_dict(torch.load(str(config['lr_scheduler']['path_to_saved_scheduler'],weights_only=True)))
        else:
          model_scheduler.load_state_dict(torch.load(dir_save_models+f"{name_model}_scheduler_state_savesafe.pt",weights_only=True))
      else:
        model_scheduler.load_state_dict(torch.load(dir_save_models+f"{name_model}_scheduler_state_savesafe.pt",weights_only=True))
    
####----------------------Training Loop-----------------------------------
for epoch in range(n_epochs):

  ####----------------------loops-----------------------------------
  calculate_parameters(spectrovit)
  train_loss = run_train_epoch(model=spectrovit, optimizer=optimizer, criterion=loss, loader=dataloader_train, epoch=epoch, device=device)
  val_loss, loader_mean_mse, loader_mean_snr,loader_mean_linewidth,loader_mean_shape_score,score_challenge = run_validation(model=spectrovit, criterion=loss, loader=dataloader_validation, epoch=epoch, device=device)

  train_loss_list.append(train_loss)
  val_loss_list.append(val_loss)
  val_mean_mse_list.append(loader_mean_mse)
  val_mean_snr_list.append(loader_mean_snr)
  val_mean_linewidth_list.append(loader_mean_linewidth)
  val_mean_shape_score_list.append(loader_mean_shape_score)
  score_challenge_list.append(score_challenge)


  ###------------------------------------------savings----------------------------------
  if save_best_model_bool is True:
     best_model(current_score=score_challenge, name_model=name_model, model=spectrovit,
                 epoch=epoch, use_wandb=use_wandb)
     
  if epoch%step_for_saving_plots == 0:
    valid_on_the_fly(model=spectrovit, epoch=epoch, val_dataset=dataset_validation, save_dir_path=save_dir_path, filename=name_model, device=device)

  if epoch%step_to_safe_save_models == 0:
    current_lr = model_scheduler.scheduler.get_last_lr()[0] if use_lr_scheduler else initial_lr
    safe_save(dir_save_models=dir_save_models, name_model=name_model,
              model=spectrovit, epoch=epoch, optimizer=optimizer, current_lr=current_lr,
              lr_scheduler=model_scheduler)
  if (epoch % step_to_safe_save_models == 0) or (epoch == n_epochs-1):
    save_losses_and_metrics(train_loss_list=train_loss_list,
                            val_loss_list=val_loss_list,
                            val_mean_mse_list=val_mean_mse_list,
                            val_mean_snr_list=val_mean_mse_list,
                            val_mean_linewidth_list=val_mean_linewidth_list,
                            val_mean_shape_score_list=val_mean_shape_score_list,
                            score_challenge_list=score_challenge_list)

  ###------------------------------------------learning rate update----------------------------------
  if use_lr_scheduler is True:
      if epoch == epoch_to_switch_to_lr_scheduler:
        for param_group in optimizer.param_groups:
          param_group['lr'] = initial_lr_scheduler
      elif epoch > epoch_to_switch_to_lr_scheduler:
        model_scheduler.step()
        print("Current learning rate:",model_scheduler.scheduler.get_last_lr()[0])

####----------------------Finishing-----------------------------------
save_trained_model(dir_save_models=dir_save_models, name_model=name_model, model=spectrovit)
delete_safe_save(dir_save_models=dir_save_models, name_model=name_model)

if use_wandb is True:
    wandb.finish()

if new_model is True:
  plot_training_evolution(path=save_dir_path,
                          train_loss_list=train_loss_list,
                          val_loss_list=val_loss_list)
else:
  if os.path.isfile(save_dir_path+'losses_and_metrics.csv'):
    metrics = retrieve_metrics_from_csv(path_file=save_dir_path+'losses_and_metrics.csv')
    plot_training_evolution(path=save_dir_path,
                          train_loss_list=metrics["LossTrain"],
                          val_loss_list=metrics["LossVal"])


  