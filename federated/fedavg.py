import torch
import numpy as np
from privacy.homomorphic import HomomorphicEncryption
from privacy.secure_aggregation import SecureAggregation

class FedAvg:
    def __init__(self, n_clients):
        """
        Initialize FedAvg aggregator
        Args:
            n_clients: number of participating clients
        """
        self.n_clients = n_clients
        # Initialize privacy mechanisms
        try:
            self.secure_agg = SecureAggregation(n_clients)
        except ImportError:
            print("Warning: SecureAggregation not available. Proceeding without secure aggregation.")
            self.secure_agg = None
            
        try:
            self.homomorphic = HomomorphicEncryption()
        except Exception as e:
            print(f"Warning: HomomorphicEncryption initialization failed. Proceeding without encryption: {e}")
            self.homomorphic = None
        
    def aggregate(self, client_parameters):
        """
        Aggregate model parameters from multiple clients
        Args:
            client_parameters: list of encrypted parameters from clients
        Returns:
            aggregated model parameters
        """
        if not client_parameters:
            return None

        # Handle dictionary format
        if isinstance(client_parameters[0], dict):
            aggregated_dict = {}
            keys = client_parameters[0].keys()
            
            for key in keys:
                key_arrays = [params[key] for params in client_parameters]
                stacked_params = np.stack(key_arrays)
                aggregated_dict[key] = np.mean(stacked_params, axis=0)
            
            return aggregated_dict
        else:
            # Decrypt parameters if they are encrypted and homomorphic is available
            decrypted_params = []
            for params in client_parameters:
                if self.homomorphic:
                    decrypted = self.homomorphic.decrypt_model_params(params)
                else:
                    decrypted = params
                decrypted_params.append(decrypted)
                
            # Remove secure aggregation masks if secure aggregation is available
            unmasked_params = []
            client_ids = list(range(self.n_clients))
            for params in decrypted_params:
                if self.secure_agg:
                    unmasked = self.secure_agg.unmask_parameters(params, client_ids)
                else:
                    unmasked = params
                unmasked_params.append(unmasked)
                
            # Average parameters
            aggregated_params = np.mean(unmasked_params, axis=0)
            
            # Re-encrypt parameters for secure distribution if homomorphic is available
            if self.homomorphic:
                encrypted_params = self.homomorphic.encrypt_model_params(aggregated_params)
                return encrypted_params
            else:
                return aggregated_params
    
    def weighted_aggregate(self, client_parameters, weights):
        """
        Aggregate model parameters with weights
        Args:
            client_parameters: list of encrypted parameters from clients
            weights: list of weights for each client
        Returns:
            weighted aggregated parameters
        """
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Handle dictionary format
        if isinstance(client_parameters[0], dict):
            aggregated_dict = {}
            keys = client_parameters[0].keys()
            
            for key in keys:
                # Extract arrays for this key from all clients
                key_arrays = [params[key] for params in client_parameters]
                # Stack and compute weighted average
                stacked_params = np.stack(key_arrays)
                aggregated_dict[key] = np.average(stacked_params, weights=weights, axis=0)
            
            return aggregated_dict
        else:
            # Decrypt and unmask parameters
            decrypted_params = []
            for params in client_parameters:
                if self.homomorphic:
                    decrypted = self.homomorphic.decrypt_model_params(params)
                else:
                    decrypted = params
                decrypted_params.append(decrypted)
                
            unmasked_params = []
            client_ids = list(range(self.n_clients))
            for params in decrypted_params:
                if self.secure_agg:
                    unmasked = self.secure_agg.unmask_parameters(params, client_ids)
                else:
                    unmasked = params
                unmasked_params.append(unmasked)
                
            # Weighted average
            weighted_params = np.average(unmasked_params, weights=weights, axis=0)
            
            # Re-encrypt parameters
            if self.homomorphic:
                encrypted_params = self.homomorphic.encrypt_model_params(weighted_params)
                return encrypted_params
            else:
                return weighted_params