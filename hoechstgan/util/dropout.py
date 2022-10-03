from torch import nn
from torch.nn.functional import dropout


class Dropout(nn.Module):
    def __init__(self,
                 p: float,
                 eval_mode: str = "identity"  # identity | dropout | average
                 ):
        super().__init__()
        self.p = p
        self.eval_mode = eval_mode

    def forward(self, x):
        if self.training or self.eval_mode == "dropout":
            return dropout(x, self.p, training=True)
        elif self.eval_mode == "identity":
            return x
        elif self.eval_mode == "average":
            return x * self.p
        else:
            raise ValueError(f"Invalid eval_mode: {self.eval_mode}")
