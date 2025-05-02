import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader

from data_preprocessing.dataset import MINDDataset
from data_preprocessing.preprocessor import NewsPreprocessor
from data_preprocessing.tokenizer import NewsTokenizer
from models.bert_encoder import BertEncoder
from models.user_encoder import UserEncoder
from models.nrms import NRMS
from federated.client import FederatedClient
from federated.server import FederatedServer
from evaluation.metrics import evaluate_model
from evaluation.privacy_metrics import compute_epsilon, estimate_privacy_risk

def main():
    # Initialize data preprocessing
    dataset = MINDDataset(
        behaviors_path='D:/SEM6/FL/project/Assignment/data/behaviors.csv',
        news_path='D:/SEM6/FL/project/Assignment/data/news.csv'
    )
    preprocessor = NewsPreprocessor()
    tokenizer = NewsTokenizer()
    
    # Create data loaders
    train_data = dataset  # In a real application, you would split dataset into train/val/test
    test_data = dataset   # For simplicity, we're using the same dataset
    
    # For simplicity, we'll create a dummy data loader
    # In a real application, you would create proper data loaders
    batch_size = 32
    
    def dummy_batch():
        # Create a dummy batch of data
        return {
            'user_news_features': {
                'input_ids': torch.randint(0, 30000, (batch_size, 50, 30)),
                'attention_mask': torch.ones(batch_size, 50, 30)
            },
            'candidate_news_features': {
                'input_ids': torch.randint(0, 30000, (batch_size, 30)),
                'attention_mask': torch.ones(batch_size, 30)
            },
            'labels': torch.randint(0, 2, (batch_size,)).float()
        }
    
    train_loader = [dummy_batch() for _ in range(10)]  # 10 batches
    val_loader = [dummy_batch() for _ in range(5)]     # 5 batches
    test_loader = [dummy_batch() for _ in range(5)]    # 5 batches
    
    # Initialize model components
    news_encoder = BertEncoder()
    user_encoder = UserEncoder()
    model = NRMS(news_encoder, user_encoder)
    
    # Setup federated learning
    n_clients = 5  
    dp_params = {
        'noise_multiplier': 1.0,
        'l2_norm_clip': 1.0,
        'delta': 1e-5
    }
    
    # Initialize server
    server = FederatedServer(model, n_clients, dp_params)
    
    # Initialize clients
    clients = []
    for i in range(n_clients):
        client_model = NRMS(BertEncoder(), UserEncoder())
        client = FederatedClient(i, client_model, dp_params)
        clients.append(client)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create a DataFrame to store training metrics
    training_metrics = []
    
    # Training loop
    n_rounds = 10
    for round_idx in range(n_rounds):
        print(f'Round {round_idx + 1}/{n_rounds}')
        
        # Distribute model to clients
        global_params = server.distribute_model()
        
        # Client training
        client_parameters = []
        client_metrics = {}
        
        # Process each client individually
        for client_idx, client in enumerate(clients):
            # Update client model
            client.update_model_parameters(global_params)
            
            # Train on local data
            client.train(train_loader, epochs=1)
            
            # Get updated parameters
            parameters = client.get_model_parameters()
            client_parameters.append(parameters)
            
            # Evaluate client performance
            metrics = evaluate_model(client.model, val_loader)
            client_metrics[client_idx] = metrics['auc']
            print(f'Client {client_idx} AUC: {metrics["auc"]:.4f}')
            
            # Save client metrics
            training_metrics.append({
                'round': round_idx + 1,
                'client_id': client_idx,
                'epsilon': compute_epsilon(dp_params['noise_multiplier'], batch_size / len(dataset), 
                                        (round_idx + 1) * len(train_loader), dp_params['delta']),
                'noise_multiplier': dp_params['noise_multiplier'],
                'accuracy': metrics['auc'],
                'loss': 1.0 - metrics['auc'],  # Simulated loss for demonstration
            })
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Update server model
        server.update_client_weights(client_metrics)
        server.aggregate_models(client_parameters)
        
        # Evaluate global model
        metrics = evaluate_model(server.model, test_loader)
        print(f'Global model AUC: {metrics["auc"]:.4f}, MRR: {metrics["mrr"]:.4f}, NDCG@5: {metrics["ndcg@5"]:.4f}')
        
        # Compute privacy budget spent
        sample_rate = batch_size / len(dataset)
        iterations = (round_idx + 1) * len(train_loader)
        epsilon = compute_epsilon(dp_params['noise_multiplier'], sample_rate, iterations, dp_params['delta'])
        privacy_risk = estimate_privacy_risk(epsilon, dp_params['delta'])
        print(f'Privacy budget spent: ε = {epsilon:.2f}, Risk = {privacy_risk:.4f}')
        
        # Save model checkpoint
        model_dir = os.path.join('models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save latest model
        torch.save(server.model.state_dict(), os.path.join(model_dir, 'latest_model.pt'))
        
        # Save checkpoint for this round
        torch.save({
            'round': round_idx + 1,
            'model_state_dict': server.model.state_dict(),
            'client_weights': server.client_weights,
            'epsilon': epsilon,
            'privacy_risk': privacy_risk,
            'metrics': metrics,
        }, os.path.join(model_dir, f'checkpoint_round_{round_idx+1}.pt'))
    
    # Save all training metrics to CSV
    metrics_df = pd.DataFrame(training_metrics)
    metrics_df.to_csv('results/experiment_results.csv', index=False)
    print("Training complete. Results saved to results/experiment_results.csv")

if __name__ == '__main__':
    main()