import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from random import shuffle

class Loss(nn.Module):
    """Custom loss function for survival analysis."""
    def __init__(self, trade_off=0.3, mode='total'):
        """
        Parameters
        ----------
        trade_off: float (Default:0.3)
            To balance the unsupervised loss and cox loss.

        mode: str (Default:'total')
            To determine which loss is used.
        """
        super(Loss, self).__init__()
        self.trade_off = trade_off
        self.mode = mode

    def cox_loss(self, pred_hazard, event, time):
        """
        Computes the negative log-likelihood for survival analysis.

        Parameters
        ----------
        pred_hazard : torch.Tensor
            The predicted hazard (risk) values from the model.

        event : torch.Tensor
            A tensor indicating whether the event occurred (1) or was censored (0).

        time : torch.Tensor
            The observed times of the events or censoring.

        Returns
        -------
        torch.Tensor
            The negative log-likelihood loss.
        """
        risk = pred_hazard
        _, idx = torch.sort(time, descending=True)
        event = event[idx]
        risk = risk[idx].squeeze()

        hazard_ratio = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-6)
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * event

        num_observed_events = torch.sum(event) + 1e-6
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events

        return neg_likelihood

    def forward(self, pred_hazard, event, time):
        """
        Forward pass to calculate the loss.

        Parameters
        ----------
        pred_hazard : torch.Tensor
            The predicted hazard (risk) values from the model.

        event : torch.Tensor
            A tensor indicating whether the event occurred (1) or was censored (0).

        time : torch.Tensor
            The observed times of the events or censoring.

        Returns
        -------
        torch.Tensor
            The calculated loss value.
        """
        loss = self.cox_loss(pred_hazard, event, time)
        return loss

