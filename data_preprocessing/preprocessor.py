import torch
import numpy as np
from transformers import BertTokenizer

class NewsPreprocessor:
    def __init__(self, max_title_length=30):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_title_length = max_title_length
        
    def preprocess_news(self, news_df):
        """
        Preprocess news articles
        Args:
            news_df: pandas DataFrame containing news articles
        Returns:
            dict: preprocessed news features
        """
        titles = news_df['title'].fillna('').values
        
        # Tokenize titles
        encoded_titles = self.tokenizer(
            titles.tolist(),
            max_length=self.max_title_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded_titles['input_ids'],
            'attention_mask': encoded_titles['attention_mask']
        }
    
    def preprocess_user_history(self, history_news_indices, max_history_length=50):
        """
        Preprocess user history
        Args:
            history_news_indices: list of news indices in user history
            max_history_length: maximum number of historical news to keep
        Returns:
            torch.Tensor: padded history indices
        """
        if len(history_news_indices) > max_history_length:
            history_news_indices = history_news_indices[-max_history_length:]
        else:
            history_news_indices = history_news_indices + [0] * (max_history_length - len(history_news_indices))
            
        return torch.tensor(history_news_indices)