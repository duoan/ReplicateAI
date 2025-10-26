from torch import optim


class NoamScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(
        self, optimizer, d_model: int, warmup_steps: int = 4000, last_epoch=-1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        # lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        scale = (self.d_model**-0.5) * min(step**-0.5, step * (self.warmup_steps**-1.5))
        return [scale for _ in self.base_lrs]
