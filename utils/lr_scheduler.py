__all__ = ["NoamLR", "SlantedTriangular"]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/scheduler.py
Update date: April-26-2019

    scheduler = _Scheduler(optimizer)
    batch_num_total = 0
    for epoch in range(num_epochs):
        for batch in batches_in_epoch:
            batch_num_total += 1
            scheduler.step_batch(batch_num_total)
        scheduler.step(validation_metrics, epoch)

    Pytorch provides StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
'''
class _Scheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for i, group in enumerate(self.optimizer.param_groups):
                group.setdefault("initial_lr", group["lr"])
        self.base_values = [group["initial_lr"] for group in self.optimizer.param_groups]
        self.step(epoch=last_epoch)
        self.last_epoch = last_epoch

    def step(self, metric: float = None, epoch: int = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self.metric = metric
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = learning_rate


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/noam.py
Update date: April-26-2019'''
class NoamLR(_Scheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        super().__init__(optimizer, last_epoch=last_epoch)

    def step_batch(self, batch_num_total=None):
        if batch_num_total is None:
            self.last_epoch += 1
        else:
            self.last_epoch = batch_num_total

        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = learning_rate

    def get_values(self):
        step = max(self.last_epoch, 1)
        scale = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [scale for _ in range(len(self.base_values))]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/slanted_triangular.py
Update date: April-26-2019'''
class SlantedTriangular(_Scheduler):
    def __init__(self, optimizer, num_epochs, num_steps_per_epoch,
                 cut_frac=0.1, ratio=32, last_epoch=-1,
                 gradual_unfreezing=False, discriminative_fine_tuning=False, decay_factor=0.38):
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = gradual_unfreezing
        self.is_first_epoch = True
        self.batch_num_total_epoch_end = []

        if self.gradual_unfreezing:
            assert not optimizer.param_groups[-1]["params"], "The default group should be empty."
        if self.gradual_unfreezing or discriminative_fine_tuning:
            assert len(optimizer.param_groups) > 2, "There should at least 3 param_groups (2 + empty default group)."

        super().__init__(optimizer, last_epoch)

        if discriminative_fine_tuning:
            exponent = 0
            for i in range(len(self.base_values) - 1, -1, -1):
                param_group = optimizer.param_groups[i]
                if param_group["params"]:
                    param_group["lr"] = self.base_values[i] * decay_factor ** exponent
                    self.base_values[i] = param_group["lr"]
                    exponent += 1

        self.last_batch_num_total = -1
        self.step_batch(0)

    def step(self, metric=None, epoch=None):
        '''The method is called once when initialising before the first epoch
            and then always at the end of each epoch'''
        if len(self.batch_num_total_epoch_end) == 0:
            self.batch_num_total_epoch_end.append(0)
        else:
            self.batch_num_total_epoch_end.append(self.last_batch_num_total)

        if self.gradual_unfreezing:
            if self.is_first_epoch:
                num_layers_to_unfreeze = 1
                self.is_first_epoch = False
            else:
                num_layers_to_unfreeze = epoch + 2

            if num_layers_to_unfreeze >= len(self.optimizer.param_groups) - 1:
                print("Gradual unfreezing finished. Training all layers.")
                self.freezing_current = False
            else:
                print("Gradual unfreezing. Training only the top %d layers." % num_layers_to_unfreeze)

            for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
                for param in param_group["params"]:
                    param.requires_grad = bool(i <= num_layers_to_unfreeze)

    def step_batch(self, batch_num_total=None):
        if batch_num_total is None:
            batch_num_total = self.last_batch_num_total + 1
        self.last_batch_num_total = batch_num_total
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = learning_rate

    def get_values(self):
        if len(self.batch_num_total_epoch_end) > 1:
            actual_num_steps_per_epoch = int(
                self.batch_num_total_epoch_end[-1] / (len(self.batch_num_total_epoch_end) - 1))
        else:
            actual_num_steps_per_epoch = max(self.num_steps_per_epoch, self.last_batch_num_total)

        if self.freezing_current:
            num_steps = actual_num_steps_per_epoch
            step = min(self.last_batch_num_total - self.batch_num_total_epoch_end[-1], num_steps)
        else:
            if not self.gradual_unfreezing:
                frozen_steps = 0
            else:
                num_frozen_epochs = len(self.optimizer.param_groups) - 2
                frozen_steps = self.batch_num_total_epoch_end[num_frozen_epochs]
            num_steps = self.num_epochs * actual_num_steps_per_epoch - frozen_steps
            step = min(self.last_batch_num_total - frozen_steps, num_steps)

        cut = int(num_steps * self.cut_frac)
        prop = step / cut if step < cut else 1 - (step - cut) / (num_steps - cut)
        return [lr * (1 + prop * (self.ratio - 1)) / self.ratio for lr in self.base_values]
