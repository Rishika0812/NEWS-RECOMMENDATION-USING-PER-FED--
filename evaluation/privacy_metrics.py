import numpy as np
import math
import torch
from sklearn.metrics import roc_curve

def compute_epsilon(noise_multiplier, sample_rate, iterations, delta=1e-5):
    """
    Compute privacy budget epsilon for given DP-SGD parameters
    Args:
        noise_multiplier: noise level in DP-SGD
        sample_rate: batch sampling rate (batch_size / dataset_size)
        iterations: number of training iterations
        delta: target delta for (ε, δ)-differential privacy
    Returns:
        epsilon value
    """
    # Simple analytical calculation using moments accountant
    # This is a simplified approximation of epsilon
    c = sample_rate * np.sqrt(2 * iterations * np.log(1/delta))
    return c / noise_multiplier

def compute_membership_advantage(model, member_data, non_member_data):
    """
    Compute membership advantage as a privacy leakage metric
    Args:
        model: trained model
        member_data: data used for training
        non_member_data: data not used for training
    Returns:
        membership advantage score
    """
    model.eval()
    # Get loss for members
    member_losses = []
    for batch in member_data:
        with torch.no_grad():
            prediction = model(batch['user_news_features'], batch['candidate_news_features'])
            loss = torch.nn.BCELoss()(prediction, batch['labels'])
            member_losses.append(loss.item())
    
    # Get loss for non-members
    non_member_losses = []
    for batch in non_member_data:
        with torch.no_grad():
            prediction = model(batch['user_news_features'], batch['candidate_news_features'])
            loss = torch.nn.BCELoss()(prediction, batch['labels'])
            non_member_losses.append(loss.item())
    
    # Convert to numpy arrays
    member_losses = np.array(member_losses)
    non_member_losses = np.array(non_member_losses)
    
    # Create labels (1 for members, 0 for non-members)
    y_true = np.concatenate([np.ones_like(member_losses), np.zeros_like(non_member_losses)])
    # Use negative loss as the score (higher loss = less likely to be a member)
    y_score = np.concatenate([-member_losses, -non_member_losses])
    
    # Calculate ROC curve and find point with maximum difference from random
    fpr, tpr, _ = roc_curve(y_true, y_score)
    advantage = np.max(tpr - fpr)
    
    return advantage

def estimate_privacy_risk(epsilon, delta, attacker_prior=0.5):
    """
    Estimate privacy risk from epsilon and delta
    Args:
        epsilon: privacy budget
        delta: target delta
        attacker_prior: attacker's prior belief about membership
    Returns:
        posterior belief about membership
    """
    # Convert epsilon to privacy risk using Bayesian analysis
    odds_ratio = np.exp(epsilon)
    prior_odds = attacker_prior / (1 - attacker_prior)
    posterior_odds = odds_ratio * prior_odds
    posterior = posterior_odds / (1 + posterior_odds)
    
    # Add the contribution of delta
    posterior += delta
    
    return min(posterior, 1.0)  # Ensure probability is at most 1

def compare_client_parameters(client_parameters, norm_type='l2'):
    """
    Compare parameter divergence between clients
    Args:
        client_parameters: list of client model parameters
        norm_type: type of norm to use for comparison
    Returns:
        pairwise distance matrix
    """
    n_clients = len(client_parameters)
    distance_matrix = np.zeros((n_clients, n_clients))
    
    for i in range(n_clients):
        for j in range(i+1, n_clients):
            # Compute distance between client parameters
            if norm_type == 'l2':
                # Compute L2 distance
                squared_diff = [(p1 - p2)**2 for p1, p2 in zip(client_parameters[i], client_parameters[j])]
                distance = np.sqrt(np.sum([np.sum(d) for d in squared_diff]))
            elif norm_type == 'l1':
                # Compute L1 distance
                abs_diff = [np.abs(p1 - p2) for p1, p2 in zip(client_parameters[i], client_parameters[j])]
                distance = np.sum([np.sum(d) for d in abs_diff])
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
                
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            
    return distance_matrix
