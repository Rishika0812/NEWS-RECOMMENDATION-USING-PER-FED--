import torch
import torch.nn as nn
import torch.nn.functional as F

class NRMS(nn.Module):
    def __init__(self, news_encoder, user_encoder):
        super(NRMS, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        
    def forward(self, user_news_features, candidate_news_features):
        """
        Forward pass of NRMS model
        Args:
            user_news_features: dict or list of news features from user history
            candidate_news_features: news features of candidate news
        Returns:
            click_probability: predicted click probability
        """
        # Check if user_news_features is a dictionary or list
        if isinstance(user_news_features, dict):
            # Handle as a batch with shared structure
            # Encode all user history news at once
            history_vectors = self.news_encoder.get_news_vector(user_news_features)
            
            # Reshape if needed - assuming input_ids has shape [batch_size, max_history_length, seq_length]
            if history_vectors.dim() == 3:
                batch_size, max_history, emb_dim = history_vectors.shape
                # Already in the correct shape for user_encoder
            else:
                # If input is [batch_size * max_history, emb_dim], reshape to [batch_size, max_history, emb_dim]
                batch_size = user_news_features['input_ids'].size(0)
                max_history_length = user_news_features['input_ids'].size(1)
                history_vectors = history_vectors.view(batch_size, max_history_length, -1)
        else:
            # Original implementation for list of features
            batch_size = len(user_news_features)
            history_vectors = []
            
            for i in range(batch_size):
                news_vectors = self.news_encoder.get_news_vector(user_news_features[i])
                history_vectors.append(news_vectors)
                
            history_vectors = torch.stack(history_vectors)
        
        # Get user representation
        user_vector = self.user_encoder.get_user_vector(history_vectors)
        
        # Encode candidate news
        candidate_vectors = self.news_encoder.get_news_vector(candidate_news_features)
        
        # Calculate click probability
        click_probability = torch.bmm(
            user_vector.unsqueeze(1),
            candidate_vectors.unsqueeze(2)
        ).squeeze()
        
        return torch.sigmoid(click_probability)
    
    def get_news_embedding(self, news_features):
        """
        Get news embeddings
        Args:
            news_features: dict containing news features
        Returns:
            tensor of shape [batch_size, embedding_dim]
        """
        return self.news_encoder.get_news_vector(news_features)
    
    def get_user_embedding(self, user_news_features):
        """
        Get user embeddings
        Args:
            user_news_features: list of news features from user history
        Returns:
            tensor of shape [batch_size, embedding_dim]
        """
        # Check if user_news_features is a dictionary or list
        if isinstance(user_news_features, dict):
            # Handle as a batch with shared structure
            history_vectors = self.news_encoder.get_news_vector(user_news_features)
            
            # Reshape if needed
            if history_vectors.dim() == 3:
                batch_size, max_history, emb_dim = history_vectors.shape
                # Already in the correct shape for user_encoder
            else:
                # If input is [batch_size * max_history, emb_dim], reshape to [batch_size, max_history, emb_dim]
                batch_size = user_news_features['input_ids'].size(0)
                max_history_length = user_news_features['input_ids'].size(1)
                history_vectors = history_vectors.view(batch_size, max_history_length, -1)
            
            return self.user_encoder.get_user_vector(history_vectors)
        else:
            # Original implementation for list of features
            history_vectors = []
            for features in user_news_features:
                news_vectors = self.news_encoder.get_news_vector(features)
                history_vectors.append(news_vectors)
                
            history_vectors = torch.stack(history_vectors)
            return self.user_encoder.get_user_vector(history_vectors)