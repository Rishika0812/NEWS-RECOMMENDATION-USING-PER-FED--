import torch
import numpy as np

class DPSGD:
    def __init__(self, noise_multiplier=1.0, l2_norm_clip=1.0, delta=1e-5):
        """
        Initialize DP-SGD
        Args:
            noise_multiplier: noise scale relative to the sensitivity
            l2_norm_clip: clipping threshold for gradients
            delta: target delta for (ε, δ)-differential privacy
        """
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.delta = delta
        
    def apply_dp(self, model, batch_size):
        """
        Apply DP-SGD algorithm directly to model gradients
        Args:
            model: PyTorch model with gradients already computed
            batch_size: size of the batch used to compute gradients
        Returns:
            noisy_gradients: list of noisy gradients to apply
        """
        # Collect gradients
        gradients = []
        for p in model.parameters():
            if p.grad is not None:
                gradients.append(p.grad.clone())
            else:
                gradients.append(torch.zeros_like(p))
                
        # Compute gradient L2 norm
        total_norm = torch.norm(torch.cat([g.flatten() for g in gradients]))
        
        # Clip gradients
        clip_coef = min(self.l2_norm_clip / (total_norm + 1e-12), 1.0)
        clipped_gradients = [g * clip_coef for g in gradients]
        
        # Add calibrated noise
        noisy_gradients = []
        for g in clipped_gradients:
            noise = torch.normal(
                mean=0.0,
                std=self.noise_multiplier * self.l2_norm_clip / max(batch_size, 1),
                size=g.shape,
                device=g.device
            )
            noisy_gradients.append(g + noise)
            
        return noisy_gradients
    
    def compute_privacy_spent(self, steps, batch_size, data_size):
        """
        Compute privacy budget spent
        Args:
            steps: number of training steps
            batch_size: size of each batch
            data_size: total number of training samples
        Returns:
            epsilon value for (ε, δ)-differential privacy
        """
        q = batch_size / data_size
        epsilon = np.sqrt(2 * np.log(1.25 / self.delta)) * self.noise_multiplier * np.sqrt(steps * q)
        return epsilon
        
    # Legacy methods kept for backward compatibility 
    # These are no longer used in our implementation
    def compute_per_sample_gradients(self, model, loss_func, data_batch):
        """
        Legacy method (not used in the new implementation)
        """
        return []
        
    def clip_gradients(self, gradients):
        """
        Legacy method (not used in the new implementation)
        """
        return gradients
        
    def add_noise(self, gradients):
        """
        Legacy method (not used in the new implementation)
        """
        return gradients