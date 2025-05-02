import torch
import torch.nn as nn
import torch.amp
import numpy as np
from models.nrms import NRMS
# Import privacy mechanisms with appropriate error handling
try:
    from privacy.dp_sgd import DPSGD
    DPSGD_AVAILABLE = True
except ImportError:
    DPSGD_AVAILABLE = False
    
try:
    from privacy.secure_aggregation import SecureAggregation
    SECURE_AGG_AVAILABLE = True
except ImportError:
    SECURE_AGG_AVAILABLE = False
    
try:
    from privacy.homomorphic import HomomorphicEncryption
    HOMOMORPHIC_AVAILABLE = True
except ImportError:
    HOMOMORPHIC_AVAILABLE = False

class FederatedClient:
    def __init__(self, client_id, model, dp_params=None):
        """
        Initialize federated learning client
        Args:
            client_id: unique identifier for the client
            model: NRMS model instance
            dp_params: differential privacy parameters
        """
        self.client_id = client_id
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())
        
        # Initialize privacy mechanisms with appropriate error handling
        self.dp_sgd = None
        if dp_params and DPSGD_AVAILABLE:
            try:
                self.dp_sgd = DPSGD(**dp_params)
                print(f"DP-SGD enabled for client {client_id}")
            except Exception as e:
                print(f"Warning: DPSGD initialization failed for client {client_id}: {e}")
        
        self.secure_agg = None
        if SECURE_AGG_AVAILABLE:
            try:
                self.secure_agg = SecureAggregation(client_id)
                print(f"Secure aggregation enabled for client {client_id}")
            except Exception as e:
                print(f"Warning: SecureAggregation initialization failed for client {client_id}: {e}")
        
        self.homomorphic = None
        if HOMOMORPHIC_AVAILABLE:
            try:
                self.homomorphic = HomomorphicEncryption()
                print(f"Homomorphic encryption enabled for client {client_id}")
            except Exception as e:
                print(f"Warning: HomomorphicEncryption initialization failed for client {client_id}: {e}")
        
    def train(self, train_loader, epochs=1):
        """
        Train model on local data
        Args:
            train_loader: DataLoader for local training data
            epochs: number of local training epochs
        """
        self.model.train()
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                # Standard training (no DP)
                if not self.dp_sgd:
                    self.optimizer.zero_grad()
                    
                    # Process data without GPU transfer to avoid issues
                    user_news = batch['user_news_features']
                    candidate_news = batch['candidate_news_features']
                    labels = batch['labels']
                    
                    # Forward pass
                    click_probability = self.model(user_news, candidate_news)
                    loss = torch.nn.BCELoss()(click_probability, labels)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    continue
                
                # DP-SGD training 
                try:
                    # Get batch data
                    user_news = batch['user_news_features']
                    candidate_news = batch['candidate_news_features']
                    labels = batch['labels']
                    batch_size = labels.size(0)
                    
                    # Compute average gradient with standard backprop
                    self.optimizer.zero_grad()
                    click_probability = self.model(user_news, candidate_news)
                    loss = torch.nn.BCELoss()(click_probability, labels)
                    loss.backward()
                    
                    # Apply DP-SGD using the new simplified method
                    noisy_grads = self.dp_sgd.apply_dp(self.model, batch_size)
                    
                    # Set gradients back to model parameters
                    self.optimizer.zero_grad()
                    for p, g in zip(self.model.parameters(), noisy_grads):
                        if p.requires_grad:
                            p.grad = g
                    
                    # Update parameters
                    self.optimizer.step()
                    
                except Exception as e:
                    print(f"Warning: DP-SGD failed for client {self.client_id}, fallback to standard training: {e}")
                    # Fallback to standard training
                    self.optimizer.zero_grad()
                    click_probability = self.model(user_news, candidate_news)
                    loss = torch.nn.BCELoss()(click_probability, labels)
                    loss.backward()
                    self.optimizer.step()
                
    def get_model_parameters(self):
        """
        Get model parameters for aggregation
        Returns:
            encrypted model parameters or regular parameters
        """
        # Collect model parameters as state dictionary
        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param.data, torch.Tensor):
                state_dict[name] = param.data.cpu().numpy()
        
        # Apply secure aggregation mask if available
        if self.secure_agg:
            try:
                masked_dict = {}
                for name, param in state_dict.items():
                    masked_dict[name] = self.secure_agg.mask_parameters(param, self.client_id)
                state_dict = masked_dict
            except Exception as e:
                print(f"Warning: Secure aggregation masking failed for client {self.client_id}: {e}")
            
        # Encrypt parameters if homomorphic encryption is enabled
        if self.homomorphic:
            try:
                encrypted_dict = self.homomorphic.encrypt_model_params(state_dict)
                return encrypted_dict
            except Exception as e:
                print(f"Warning: Homomorphic encryption failed for client {self.client_id}: {e}")
        
        return state_dict
    
    def update_model_parameters(self, global_params):
        """
        Update local model with global parameters
        Args:
            global_params: global model parameters
        """
        # Decrypt parameters if they are encrypted and homomorphic encryption is enabled
        if self.homomorphic:
            try:
                params = self.homomorphic.decrypt_model_params(global_params)
            except Exception as e:
                print(f"Warning: Homomorphic decryption failed for client {self.client_id}: {e}")
                params = global_params
        else:
            params = global_params
        
        # Update model parameters
        for name, param in self.model.named_parameters():
            if name in params:
                try:
                    if isinstance(params[name], np.ndarray):
                        param.data = torch.from_numpy(params[name]).to(param.device)
                    else:
                        param.data = params[name].to(param.device)
                except Exception as e:
                    print(f"Warning: Failed to update parameter {name} for client {self.client_id}: {e}")