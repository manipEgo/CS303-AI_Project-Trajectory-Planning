import os
import time
import torch
import torch.nn as nn
from typing import Tuple
from src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate, compute_traj
from model import Net

PATH = './model.pth'
TOTAL_TIME = 0.3
RAND_TIME = 0.5 * TOTAL_TIME
RESERVED_TIME = 0.02
LEARNING_RATE = 0.5
THRESHOLD = 1

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
        traj: torch.Tensor, 
        target_pos: torch.Tensor, 
        target_scores: torch.Tensor,
        radius: float,
        ) -> torch.Tensor:
        cdist = torch.cdist(target_pos, traj)
        d = cdist.min(-1).values - radius
        d = self.relu(d)
        loss = torch.sum(d * target_scores, dim=-1)
        return loss

    def get_action(self,
        target_pos: torch.Tensor,
        target_features: torch.Tensor,
        class_scores: torch.Tensor,
        verbose: bool=False,
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
        
        start_time = time.time()

        # predict classes
        with torch.no_grad():
            outputs = self.classifier(target_features)
            _, target_classes = torch.max(outputs.data, 1)

        # rand best for some time
        ctps_inter = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS, 2.]) + torch.tensor([0., -1.])
        ctps_inter.requires_grad = True
        best_score = self.loss(compute_traj(ctps_inter), target_pos, class_scores[target_classes], RADIUS)

        # rands = []
        cnt = 0
        while time.time() - start_time < TOTAL_TIME - RESERVED_TIME:
            cnt += 1
            temp = torch.rand((N_CTPS-2, 2)) * torch.tensor([N_CTPS, 2.]) + torch.tensor([0., -1.])
            # temp.requires_grad = True
            score = evaluate(compute_traj(temp), target_pos, class_scores[target_classes], RADIUS)
            if score > best_score:
                best_score = score
                ctps_inter = temp
            # rands.append((score, temp))
        # rands.sort(key=lambda x : x[0])

        # cnt = 0
        # for rand in rands:
        #     loss = rand[0]
        #     diff = THRESHOLD
        #     opt = torch.optim.Adam([rand[1]], lr = LEARNING_RATE)
        #     while time.time() - start_time < TOTAL_TIME - RESERVED_TIME and diff >= THRESHOLD:
        #         opt.zero_grad()
        #         loss.backward()
        #         opt.step()
        #         score = loss
        #         loss = self.loss(compute_traj(rand[1]), target_pos, class_scores[target_classes], RADIUS)
        #         diff = abs(loss - score)
        #     if loss < best_score:
        #         ctps_inter = rand[1]
        #         best_score = loss
        #     cnt += 1
        #     if time.time() - start_time > TOTAL_TIME - RESERVED_TIME:
        #         break

        if verbose:
            return ctps_inter, cnt
        return ctps_inter

