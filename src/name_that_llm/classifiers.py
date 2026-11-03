import torch.nn as nn

class LogLinear(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.out = nn.Linear(d_in, n_classes)
    def forward(self, x):
        return self.out(x)  # logits; use CrossEntropyLoss

class MLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=(256,), act=nn.ReLU):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        d_prev = d_in
        for h in hidden:
            self.hidden_layers.append(nn.Linear(d_prev, h))
            d_prev = h
        self.out = nn.Linear(d_prev, n_classes)
        self.act = act()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act(layer(x))   # a = σ(W a + b)
        return self.out(x)           # logits z = W_out a + b_out