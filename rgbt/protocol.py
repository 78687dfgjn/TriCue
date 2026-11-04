"""TriCue's current protocol; historical C uses a different 400-epoch schedule."""
import math

DEFAULTS={'rgbt':dict(epochs=80,patience=0),
          'vtsar':dict(epochs=200,patience=50),
          'vtuav':dict(epochs=200,patience=50)}

class Map50Stopper:
    def __init__(self,patience=50):
        self.patience=int(patience);self.best_epoch=0;self.best_fitness=-1.
        self.metric=None;self.possible_stop=False
    def __call__(self,epoch,fitness=None):
        if self.metric is None:return False
        if not math.isfinite(self.metric):raise FloatingPointError('Nonfinite Val mAP50')
        if self.metric>self.best_fitness:
            self.best_fitness=float(self.metric);self.best_epoch=epoch
        delta=epoch-self.best_epoch
        self.possible_stop=bool(self.patience and delta>=self.patience-1)
        return bool(self.patience and delta>=self.patience)
