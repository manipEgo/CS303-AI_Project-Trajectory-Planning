import os
import time
import torch
import torch.nn as nn
from functorch import vmap
from typing import Tuple
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, compute_traj
from model import Net

PATH = os.path.join(os.path.dirname(__file__), 'model.pth')
TOTAL_TIME = 0.3
RESERVED_TIME = 0.012
LEARNING_RATE = 2e-3

RAND_TIME = 0.2 * TOTAL_TIME
RAND_NUM = 128

THRESHOLD = 5

class Agent:

    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """

        # TODO: prepare your agent here
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Net()
        print(os.getcwd())
        model.load_state_dict(torch.load(PATH, map_location=self.device))
        self.classifier = model
        self.relu = nn.ReLU()

    def loss(self,
        ctps_inter: torch.Tensor, 
        target_pos: torch.Tensor, 
        target_scores: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        traj = compute_traj(ctps_inter)
        cdist = torch.cdist(target_pos, traj)
        d = cdist.min(-1).values - radius
        loss = torch.sigmoid(self.relu(d)) * 2
        loss = torch.sum(loss * target_scores, dim=-1)
        return loss

    def get_action(self,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
        class_scores: torch.Tensor,
        verbose: bool=False,
    ):
        """Compute the parameters required to fire a projectile. 
        
        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        # assert len(target_pos) == len(target_features)
        
        start_time = time.time()

        # predict classes
        with torch.no_grad():
            outputs = self.classifier(target_features)
            _, target_classes = torch.max(outputs.data, 1)

        # rand best for some time
        ctps_inters = []
        tot = 0
        while time.time() - start_time < RAND_TIME or tot < RAND_NUM:
            temp = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS * 3, 6.]) + torch.tensor([-N_CTPS, -3.])
            temp.requires_grad = True
            ctps_inters.append(temp)
            tot += 1
        ctps_inters = torch.stack(ctps_inters)
        batch_compute_traj = vmap(compute_traj, in_dims=0, out_dims=0)
        batch_evaluate = vmap(evaluate, in_dims=(0, None, None, None), out_dims=0)
        scores = batch_evaluate(batch_compute_traj(ctps_inters), target_pos, class_scores[target_classes], RADIUS)
        idxs = torch.argsort(scores, stable=True, descending=True)
        scores = scores[idxs]
        ctps_inters = ctps_inters[idxs]
        result = ctps_inters[0]
        best_eva = scores[0]

        cnt = 0
        while time.time() - start_time < TOTAL_TIME - RESERVED_TIME and cnt < tot:
            temp = ctps_inters[cnt].clone().detach().requires_grad_(True)
            temp.requires_grad = True
            score = self.loss(temp, target_pos, class_scores[target_classes], RADIUS)
            opt = torch.optim.NAdam([temp], lr=LEARNING_RATE)
            diff = THRESHOLD
            while time.time() - start_time < TOTAL_TIME - RESERVED_TIME and diff >= THRESHOLD:
                opt.zero_grad()
                score.backward()
                opt.step()
                loss = score
                score = self.loss(temp, target_pos, class_scores[target_classes], RADIUS)
                diff = abs(loss - score)
            eva = evaluate(compute_traj(temp), target_pos, class_scores[target_classes], RADIUS)
            if eva > best_eva:
                best_eva = eva
                result = temp
            cnt += 1

        if best_eva < 0.0:
            temp_result = []
            temp_result.append(torch.tensor([-2., 1.]))
            temp_result.append(torch.tensor([2.5, 2.]))
            temp_result.append(torch.tensor([7., 1.]))

        if verbose:
            return result, cnt, tot
        return result

