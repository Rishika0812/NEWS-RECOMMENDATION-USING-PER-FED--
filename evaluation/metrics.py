import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

def compute_auc(y_true, y_pred):
    """
    Compute Area Under the ROC Curve
    Args:
        y_true: ground truth labels
        y_pred: predicted scores
    Returns:
        AUC score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        print(f"Warning: {e}")
        return 0.5  # Default value if calculation fails
        
def compute_mrr(y_true, y_pred):
    """
    Compute Mean Reciprocal Rank
    Args:
        y_true: ground truth labels
        y_pred: predicted scores
    Returns:
        MRR score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Get ranks of relevant items
    sorted_indices = np.argsort(-y_pred, axis=1)
    relevant_ranks = []
    
    for i, indices in enumerate(sorted_indices):
        for rank, idx in enumerate(indices):
            if y_true[i, idx] == 1:
                relevant_ranks.append(1.0 / (rank + 1))
                break
                
    return np.mean(relevant_ranks) if relevant_ranks else 0.0

def compute_ndcg(y_true, y_pred, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at k
    Args:
        y_true: ground truth labels
        y_pred: predicted scores
        k: top-k items to consider
    Returns:
        NDCG@k score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Sort predictions and get top-k indices
    sorted_indices = np.argsort(-y_pred, axis=1)[:, :k]
    
    ndcg_scores = []
    for i, indices in enumerate(sorted_indices):
        # Get relevance scores for top-k predictions
        relevance = np.array([y_true[i, idx] for idx in indices])
        # Calculate DCG
        dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
        
        # Calculate ideal DCG
        ideal_relevance = np.sort(y_true[i])[::-1][:k]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
        
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
        
    return np.mean(ndcg_scores)

def evaluate_model(model, data_loader):
    """
    Evaluate model on the given data loader
    Args:
        model: prediction model
        data_loader: data loader containing evaluation data
    Returns:
        dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            user_news = batch['user_news_features']
            candidate_news = batch['candidate_news_features']
            labels = batch['labels']
            
            predictions = model(user_news, candidate_news)
            
            all_preds.append(predictions)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return {
        'auc': compute_auc(all_labels, all_preds),
        'mrr': compute_mrr(all_labels, all_preds),
        'ndcg@5': compute_ndcg(all_labels, all_preds, k=5),
        'ndcg@10': compute_ndcg(all_labels, all_preds, k=10)
    }
