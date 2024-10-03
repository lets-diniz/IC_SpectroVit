"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import torch


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


class SaveCurrentModel:

    def __init__(self, dir_model="", valid_score=float("inf")):
        self.dir_model = dir_model

    def __call__(self, current_valid_score, model, name_model, wandb_run=None):
        print(f"Saving current model with score: {current_valid_score}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            f"{self.dir_model}{name_model}_safesave.pt",
        )

        if wandb_run is not None:
            wandb_run.save(f"{self.dir_model}_safesave")
