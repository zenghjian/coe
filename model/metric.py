import torch
import numpy as np
from itertools import combinations

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def off_penalty_loss(off):
    return off

def off_penalty_loss_percent(off_percentage):
    return off_percentage


def pos_contrastive_loss(contrastive):
    return contrastive

def pos_contrastive_loss_percent(contrastive_percentage):
    return contrastive_percentage


def ortho_loss(ortho):
    return ortho

def ortho_loss_percent(ortho_percentage):
    return ortho_percentage

def learning_rate(lr):
    return lr

def grad_norm_off_penalty(grad_norm):
    return grad_norm

def grad_norm_pos_contrastive(grad_norm):
    return grad_norm

def grad_norm_ortho(grad_norm):
    return grad_norm

def cos_sim(cos_sim):
    return cos_sim

def sum_grad_norm(grad_norm):
    return grad_norm