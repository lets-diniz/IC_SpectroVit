"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import torch.optim.lr_scheduler as lr_scheduler


class CustomLRScheduler:
    def __init__(self,
                 optimizer,
                 scheduler_type,
                 **kwargs):

        self.scheduler_type = scheduler_type.lower()
        self.scheduler = self._create_scheduler(optimizer, **kwargs)

    def _create_scheduler(self, optimizer, **kwargs):
        if self.scheduler_type == 'steplr':
            return lr_scheduler.StepLR(optimizer, **kwargs)

        elif self.scheduler_type == 'multisteplr':
            return lr_scheduler.MultiStepLR(optimizer, **kwargs)

        elif self.scheduler_type == 'exponentiallr':
            return lr_scheduler.ExponentialLR(optimizer, **kwargs)

        elif self.scheduler_type == 'cosineannealingwarmrestartlr':
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)

        elif self.scheduler_type == 'cosineannealinglr':
            return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

        elif self.scheduler_type == 'reducelronplateau':
            return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

        elif self.scheduler_type == 'cycliclr':
            return lr_scheduler.CyclicLR(optimizer, **kwargs)

        elif self.scheduler_type == 'onecyclelr':
            return lr_scheduler.OneCycleLR(optimizer, **kwargs)

        else:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")

    def step(self, *args, **kwargs):
        if self.scheduler_type == 'reducelronplateau':
            self.scheduler.step(*args, **kwargs)
        else:
            self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
