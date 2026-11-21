import torch.nn as nn

from name_that_llm.config import DEFAULT_MLP, MLPConfig, TrainingConfig
from name_that_llm.lstm import LSTM

MODELS = ("log-linear", "mlp", "svm")


class LogLinear(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.out = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.out(x)  # logits; use CrossEntropyLoss


class MLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=DEFAULT_MLP.hidden, act=None):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        d_prev = d_in
        for h in hidden:
            self.hidden_layers.append(nn.Linear(d_prev, h))
            d_prev = h
        self.out = nn.Linear(d_prev, n_classes)
        self.act = (getattr(nn, DEFAULT_MLP.activation) if act is None else act)()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act(layer(x))  # a = σ(W a + b)
        return self.out(x)  # logits z = W_out a + b_out


def build_model(
    config: TrainingConfig, mlp_config: MLPConfig, d_in: int, n_classes: int
) -> nn.Module:
    use_lstm = config.pooling == "lstm"
    feature_size = config.lstm_hidden_size if use_lstm else d_in
    if config.model in ("log-linear", "svm"):
        # A linear SVM is the same linear layer as log-linear, trained with a hinge loss.
        classifier = LogLinear(feature_size, n_classes)
    else:
        classifier = MLP(
            feature_size,
            n_classes,
            hidden=mlp_config.hidden,
            act=getattr(nn, mlp_config.activation),
        )
    if use_lstm:
        return nn.Sequential(
            LSTM(d_in, config.lstm_hidden_size, normalize=config.normalize),
            classifier,
        )
    return classifier


def build_loss(config: TrainingConfig) -> nn.Module:
    """Multi-class hinge loss for the SVM; cross-entropy for the probabilistic models."""
    return nn.MultiMarginLoss() if config.model == "svm" else nn.CrossEntropyLoss()
