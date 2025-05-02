import torch
import numpy as np
from models.nrms import NRMS
from federated.fedavg import FedAvg
from evaluation.metrics import evaluate_model

# Import privacy mechanisms with appropriate error handling
try:
    from privacy.homomorphic import HomomorphicEncryption
    HOMOMORPHIC_AVAILABLE = True
except ImportError:
    HOMOMORPHIC_AVAILABLE = False
    
try:
    from privacy.secure_aggregation import SecureAggregation
    SECURE_AGG_AVAILABLE = True
except ImportError:
    SECURE_AGG_AVAILABLE = False

class FederatedServer:
    def __init__(self, model, n_clients, dp_params=None):
        """
        Initialize federated learning server
        Args:
            model: global NRMS model instance
            n_clients: number of participating clients
            dp_params: differential privacy parameters
        """
        self.model = model
        self.n_clients = n_clients
        self.dp_params = dp_params
        
        # Initialize aggregation mechanisms
        self.fedavg = FedAvg(n_clients)
        
        # Initialize secure aggregation if available
        self.secure_agg = None
        if SECURE_AGG_AVAILABLE:
            try:
                self.secure_agg = SecureAggregation(n_clients)
                print("Secure aggregation enabled on server")
            except Exception as e:
                print(f"Warning: SecureAggregation initialization failed on server: {e}")
        
        # Initialize homomorphic encryption if available
        self.homomorphic = None
        if HOMOMORPHIC_AVAILABLE:
            try:
                self.homomorphic = HomomorphicEncryption()
                print("Homomorphic encryption enabled on server")
            except Exception as e:
                print(f"Warning: HomomorphicEncryption initialization failed on server: {e}")
        
        # Store client contributions for weighted aggregation
        self.client_weights = np.ones(n_clients) / n_clients
        
    def distribute_model(self):
        """
        Prepare model for distribution to clients
        Returns:
            encrypted model parameters
        """
        # Create a state dictionary from model parameters
        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param.data, torch.Tensor):
                state_dict[name] = param.data.clone().cpu().numpy()
                
        # Encrypt parameters if homomorphic encryption is available
        if self.homomorphic:
            try:
                encrypted_dict = self.homomorphic.encrypt_model_params(state_dict)
                return encrypted_dict
            except Exception as e:
                print(f"Warning: Homomorphic encryption failed on server: {e}")
                
        return state_dict
    
    def aggregate_models(self, client_parameters, weighted=True):
        """
        Aggregate model parameters from clients
        Args:
            client_parameters: list of encrypted parameters from clients
            weighted: whether to use weighted aggregation
        """
        if not client_parameters:
            print("Warning: No client parameters to aggregate")
            return
            
        try:
            if weighted:
                aggregated_params = self.fedavg.weighted_aggregate(
                    client_parameters,
                    self.client_weights
                )
            else:
                aggregated_params = self.fedavg.aggregate(client_parameters)
                
            # Update global model
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    if isinstance(aggregated_params[name], np.ndarray):
                        param.data = torch.from_numpy(aggregated_params[name]).to(param.device)
                    else:
                        param.data = aggregated_params[name].to(param.device)
        except Exception as e:
            print(f"Warning: Model aggregation failed: {e}")
    
    def update_client_weights(self, client_metrics):
        """
        Update client weights based on their performance
        Args:
            client_metrics: dictionary of client performance metrics
        """
        if not client_metrics:
            print("Warning: No client metrics to update weights")
            return
            
        try:
            # Update weights based on validation performance
            total_score = sum(client_metrics.values())
            if total_score > 0:
                for client_id, score in client_metrics.items():
                    self.client_weights[client_id] = score / total_score
            
            # Normalize weights
            self.client_weights = self.client_weights / np.sum(self.client_weights)
        except Exception as e:
            print(f"Warning: Failed to update client weights: {e}")
            # Reset to equal weights
            self.client_weights = np.ones(self.n_clients) / self.n_clients
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model performance
        Args:
            test_loader: DataLoader for test data
        Returns:
            evaluation metrics
        """
        return evaluate_model(self.model, test_loader)