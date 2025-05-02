import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecureAggregation:
    def __init__(self, n_clients):
        """
        Initialize secure aggregation protocol
        Args:
            n_clients: number of participating clients
        """
        self.n_clients = n_clients
        self.keys = [Fernet.generate_key() for _ in range(n_clients)]
        self.ferners = [Fernet(key) for key in self.keys]
        
    def generate_mask(self, params_shape, client_id):
        """
        Generate random mask for a client
        Args:
            params_shape: shape of model parameters
            client_id: ID of the client
        Returns:
            random mask array
        """
        np.random.seed(client_id)
        return np.random.normal(0, 0.01, params_shape)
    
    def mask_parameters(self, params, client_id):
        """
        Mask model parameters
        Args:
            params: model parameters to mask
            client_id: ID of the client
        Returns:
            masked parameters
        """
        mask = self.generate_mask(params.shape, client_id)
        return params + mask
    
    def unmask_parameters(self, masked_params, client_ids):
        """
        Remove masks from aggregated parameters
        Args:
            masked_params: aggregated masked parameters
            client_ids: list of client IDs
        Returns:
            unmasked parameters
        """
        total_mask = np.zeros_like(masked_params)
        for client_id in client_ids:
            mask = self.generate_mask(masked_params.shape, client_id)
            total_mask += mask
            
        return masked_params - total_mask
    
    def encrypt_parameters(self, params, client_id):
        """
        Encrypt parameters for secure transmission
        Args:
            params: parameters to encrypt
            client_id: ID of the client
        Returns:
            encrypted parameters
        """
        params_bytes = params.tobytes()
        return self.ferners[client_id].encrypt(params_bytes)
    
    def decrypt_parameters(self, encrypted_params, client_id):
        """
        Decrypt parameters
        Args:
            encrypted_params: encrypted parameters
            client_id: ID of the client
        Returns:
            decrypted parameters
        """
        decrypted_bytes = self.ferners[client_id].decrypt(encrypted_params)
        return np.frombuffer(decrypted_bytes, dtype=np.float32)