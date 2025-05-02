import numpy as np
import torch
from typing import List, Union

class HomomorphicEncryption:
    def __init__(self, n_bits=2048):
        """
        Initialize homomorphic encryption system using a simplified approach
        Args:
            n_bits: number of bits for the encryption key (kept for compatibility)
        """
        self.enabled = True
        self._key = np.random.randint(1, 1000, size=1)[0]  # Simple multiplicative key
        
    def encrypt_model_params(self, params: Union[np.ndarray, dict, torch.nn.Module]) -> Union[np.ndarray, dict]:
        """
        Encrypt model parameters using simple multiplicative masking
        Args:
            params: numpy array, dictionary, or PyTorch model parameters
        Returns:
            encrypted parameters in the same format as input
        """
        if not self.enabled:
            return params
            
        # Convert PyTorch model to state dict if needed
        if isinstance(params, torch.nn.Module):
            params = params.state_dict()
            
        if isinstance(params, dict):
            encrypted_dict = {}
            for key, value in params.items():
                if isinstance(value, torch.Tensor):
                    value_array = value.detach().cpu().numpy().astype(np.float64)
                elif isinstance(value, np.ndarray):
                    value_array = value.astype(np.float64)
                else:
                    value_array = np.array(value, dtype=np.float64)
                scaled_params = (value_array * 1e6)
                noise = np.random.normal(0, 1e-6, size=scaled_params.shape)
                encrypted_dict[key] = (scaled_params + noise) * self._key
            return encrypted_dict
        elif isinstance(params, torch.Tensor):
            params_array = params.detach().cpu().numpy().astype(np.float64)
            scaled_params = (params_array * 1e6)
            noise = np.random.normal(0, 1e-6, size=scaled_params.shape)
            return (scaled_params + noise) * self._key
        elif isinstance(params, np.ndarray):
            params_array = params.astype(np.float64)
            scaled_params = (params_array * 1e6)
            noise = np.random.normal(0, 1e-6, size=scaled_params.shape)
            return (scaled_params + noise) * self._key
        else:
            raise ValueError("Unsupported parameter type. Must be PyTorch model, tensor, numpy array, or dictionary.")
    
    def decrypt_model_params(self, encrypted_params: Union[np.ndarray, dict]) -> Union[np.ndarray, dict, torch.Tensor]:
        """
        Decrypt model parameters
        Args:
            encrypted_params: encrypted parameters array or dictionary
        Returns:
            decrypted parameters in the same format as input
        """
        if not self.enabled:
            return encrypted_params
            
        if isinstance(encrypted_params, dict):
            decrypted_dict = {}
            for key, value in encrypted_params.items():
                decrypted_params = value / self._key
                decrypted_array = (decrypted_params / 1e6).astype(np.float32)
                # Always convert back to PyTorch tensor for model parameters
                decrypted_dict[key] = torch.from_numpy(decrypted_array)
            return decrypted_dict
        else:
            decrypted_params = encrypted_params / self._key
            decrypted_array = (decrypted_params / 1e6).astype(np.float32)
            return torch.from_numpy(decrypted_array) if isinstance(encrypted_params, torch.Tensor) else decrypted_array
    
    def aggregate_encrypted_params(self, encrypted_params_list: List[Union[np.ndarray, dict]]) -> Union[np.ndarray, dict]:
        """
        Aggregate encrypted parameters from multiple clients
        Args:
            encrypted_params_list: list of encrypted parameters from different clients
        Returns:
            aggregated encrypted parameters
        """
        if not encrypted_params_list:
            return None
            
        # Handle dictionary format
        if isinstance(encrypted_params_list[0], dict):
            aggregated_dict = {}
            # Get all keys from the first dictionary
            keys = encrypted_params_list[0].keys()
            
            for key in keys:
                # Extract arrays for this key from all clients
                key_arrays = [params[key] for params in encrypted_params_list]
                # Stack and compute mean
                stacked_params = np.stack(key_arrays)
                aggregated_dict[key] = np.mean(stacked_params, axis=0)
            
            return aggregated_dict
        else:
            # Handle numpy array format
            stacked_params = np.stack(encrypted_params_list)
            return np.mean(stacked_params, axis=0)