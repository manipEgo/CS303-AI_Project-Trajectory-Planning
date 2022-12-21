import os
import time
import torch
import torch.nn as nn
from typing import Tuple
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, compute_traj
from model import Net

PATH = './model.pth'
RESERVED_TIME = 0.02

class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        # TODO: prepare your agent here
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Net()
        print(os.getcwd())
        model.load_state_dict(torch.load(PATH, map_location=self.device))
        self.classifier = model

    def get_action(self,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
        class_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile. 
        
        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        # assert len(target_pos) == len(target_features)
        
        # TODO: compute the firing speed and angle that would give the best score.
        start_time = time.time()
        with torch.no_grad():
            outputs = self.classifier(target_features)
            _, target_classes = torch.max(outputs.data, 1)

        # Example: return a random configuration
        ctps_inter = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
        best_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_classes], RADIUS)
        while time.time() - start_time < 0.3 - RESERVED_TIME:
            temp = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS-2, 2.]) + torch.tensor([1., -1.])
            score = evaluate(compute_traj(temp), target_pos, class_scores[target_classes], RADIUS)
            if score > best_score:
                ctps_inter = temp
                best_score = score

        return ctps_inter

