import torch

class SaveBestModel:

    def __init__(
            self, dir_model='', best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.dir_model = dir_model

    def __call__(
            self, current_valid_loss, model, name_model, run=None
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation score: {self.best_valid_loss}")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f'{self.dir_model}{name_model}')

            if run is not None:
                run.save(f'{self.dir_model}{name_model}')
