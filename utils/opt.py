import torch


'''Reference url: http://nlp.seas.harvard.edu/2018/04/03/attention.html
Update date: April-15-2019'''
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self._step = 0
        self._rate = 0
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer

    def rate(self, step=None):
        if step is None:
            step = self._step
        rate = min(step ** (-0.5), step * self.warmup ** (-1.5))
        rate *= self.model_size ** (-0.5)
        return self.factor * rate

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    @classmethod
    def get_std_opt(cls, model, factor=2, warmup=4000):
        model_size = model.src_embed[0].get_output_dim()
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        return NoamOpt(model_size, factor, warmup, optimizer)