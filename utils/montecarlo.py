import torch
import numpy as np
from typing import Union, List

class MonteCarloDropout(object):
    def __init__(self,
                 size: Union[List, int],
                 passes: int = 10,
                 classes: int = 19):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vanilla_prediction = torch.zeros(size=(1, size[0], size[1]), device=self.device)
        self.vanilla_confidence = torch.zeros(size=(1, size[0], size[1]), device=self.device)
        self.mcd_predictions = torch.zeros(size=(passes, size[0], size[1]), device=self.device)
        self.mcd_confidences = torch.zeros(size=(passes, size[0], size[1]), device=self.device)
        self.softmax = torch.zeros(size=(passes, classes, size[0], size[1]), device=self.device)
        self.mean_softmax = None
        self.var_softmax = None
        self.passes = passes

        # Save Dropout layers for checking
        self.dropout_layers = []

    def enable_dropout(self,
                       model: torch.nn.Module):
        """

        Args:
            model: Pytorch model

        """
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
                self.dropout_layers.append(m)

    def save_predictions(self,
                         pass_idx: int,
                         current_prediction: torch.Tensor,
                         current_confidence: torch.Tensor):

        if type(current_prediction) == torch.Tensor:

            # Send tensors to CPU and convert to numpy
            current_prediction = current_prediction.squeeze(0).cpu().numpy()
            current_confidence = current_confidence.squeeze(0).cpu().numpy()

        self.mcd_predictions[pass_idx] = current_prediction
        self.mcd_confidences[pass_idx] = current_confidence

    def save_softmax(self,
                     pass_idx: int,
                     softmax: torch.Tensor):
        self.softmax[pass_idx] = softmax

    def avg_softmax(self):
        # Average softmax over all forward passes
        self.mean_softmax = torch.mean(self.softmax, dim=0, keepdim=True)
        self.var_softmax = torch.var(self.softmax, dim=0, keepdim=True)

        # Get mean confidence and prediction
        confidence, prediction = self.mean_softmax.max(dim=1)

        return confidence, prediction, self.mean_softmax

    def avg_predictions(self):

        # Calculate mean and var over multiple MCD predictions
        mean_pred = np.mean(self.mcd_predictions, axis=0)
        var_pred = np.var(self.mcd_predictions, axis=0)

        # Calculate mean and var over multiple MCD confidences
        mean_conf = np.mean(self.mcd_confidences, axis=0)
        var_conf = np.var(self.mcd_confidences, axis=0)

        return {"Mean prediction": mean_pred,
                "Variance prediction": var_pred,
                "Mean confidence": mean_conf,
                "Variance confidence": var_conf}
