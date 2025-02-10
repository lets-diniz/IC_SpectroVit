"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import torch
import json
import csv
import wandb
import os

class SaveBestModel:
    def __init__(self, dir_model="", best_valid_score=float("inf")):
        self.best_valid_score = best_valid_score
        self.dir_model = dir_model

    def __call__(self, current_valid_score, model, name_model, wandb_run=None):
        if current_valid_score < self.best_valid_score:
            self.best_valid_score = current_valid_score
            print(f"Best validation score: {self.best_valid_score}")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                f"{self.dir_model}{name_model}.pt",
            )

            if wandb_run is not None:
                wandb_run.save(f"{self.dir_model}{name_model}")

#same as previous function, but also saves current epoch in a JSON file.
class SaveBestModelState:
    def __init__(self, dir_save_model, best_score=float("inf")):
        self.best_score = best_score
        self.dir_save_model = dir_save_model

    def __call__(self, current_score, name_model, model,
                 epoch, use_wandb=None):
        if current_score < self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), f"{self.dir_save_model}{name_model}_best.pt")
            training_state = {
                'epoch': epoch+1,
                'best_score': self.best_score
            }
            with open(f"{self.dir_save_model}{name_model}_best_info", "w") as outfile:
                outfile.write(json.dumps(training_state, indent=4))
            if use_wandb is True:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "_best.pt"))

class SaveLossesAndMetrics:
    def __init__(self, dir_save_results, save_count_idx=0):
        self.save_count_idx = save_count_idx
        self.dir_save_results = dir_save_results
    def __call__(self, train_loss_list,
                        val_loss_list,
                        val_mean_mse_list,
                        val_mean_snr_list,
                        val_mean_linewidth_list,
                        val_mean_shape_score_list,
                        score_challenge_list):
        with open(self.dir_save_results+'losses_and_metrics.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for i in range(self.save_count_idx, len(train_loss_list)):
                writer.writerow([train_loss_list[i],
                                 val_loss_list[i],
                                 val_mean_mse_list[i],
                                 val_mean_snr_list[i],
                                 val_mean_linewidth_list[i],
                                 val_mean_shape_score_list[i],
                                 score_challenge_list[i]])
        self.save_count_idx = len(train_loss_list)

#save current model, optimizer and learning rate to resume training 
#in case something goes wrong (power loss, for example)
def safe_save(dir_save_models, name_model,
              model, epoch, optimizer, current_lr,
              lr_scheduler=None):
    torch.save(model.state_dict(),
               f"{dir_save_models}{name_model}_savesafe.pt")
    torch.save(optimizer.state_dict(),
               f"{dir_save_models}{name_model}_optimizer_savesafe.pt")
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(),
                f"{dir_save_models}{name_model}_scheduler_state_savesafe.pt")
    training_state = {
        'epoch': epoch+1,
        'learning_rate': current_lr
    }
    with open(f"{dir_save_models}{name_model}_training_state_savesafe", "w") as outfile:
        outfile.write(json.dumps(training_state, indent=4))

#delete files used to resume training to save space
def delete_safe_save(dir_save_models, name_model):
    if os.path.isfile(f"{dir_save_models}{name_model}_trained.pt"):
        if os.path.isfile(f"{dir_save_models}{name_model}_savesafe.pt"):
            os.remove(f"{dir_save_models}{name_model}_savesafe.pt")
        if os.path.isfile(f"{dir_save_models}{name_model}_optimizer_savesafe.pt"):
            os.remove(f"{dir_save_models}{name_model}_optimizer_savesafe.pt")
        if os.path.isfile(f"{dir_save_models}{name_model}_scheduler_state_savesafe.pt"):
            os.remove(f"{dir_save_models}{name_model}_scheduler_state_savesafe.pt")
        if os.path.isfile(f"{dir_save_models}{name_model}_training_state_savesafe"):
            os.remove(f"{dir_save_models}{name_model}_training_state_savesafe")

#save final model
def save_trained_model(dir_save_models, name_model, model):
    torch.save(model.state_dict(), f"{dir_save_models}{name_model}_trained.pt")

