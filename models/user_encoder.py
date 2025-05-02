import torch
import torch.nn as nn
import torch.nn.functional as F

class UserEncoder(nn.Module):
    def __init__(self, news_embedding_dim=768, attention_hidden_dim=200):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_embedding_dim
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(news_embedding_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, news_vectors):
        """
        Encode user representation from historical news interactions
        Args:
            news_vectors: tensor of shape [batch_size, max_history_length, news_embedding_dim]
        Returns:
            tensor of shape [batch_size, news_embedding_dim]
        """
        # Calculate attention weights
        attention_weights = self.attention(news_vectors)
        
        # Apply attention weights to get user embedding
        user_vector = torch.sum(attention_weights * news_vectors, dim=1)
        
        return user_vector
    
    def get_user_vector(self, news_vectors):
        """
        Get user vector for a batch of users
        Args:
            news_vectors: tensor of shape [batch_size, max_history_length, news_embedding_dim]
        Returns:
            tensor of shape [batch_size, news_embedding_dim]
        """
        return self.forward(news_vectors)