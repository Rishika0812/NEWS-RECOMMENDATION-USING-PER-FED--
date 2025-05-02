import torch
import torch.nn as nn
from transformers import BertModel

class BertEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', dropout=0.2):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        
        # Freeze BERT parameters for efficiency
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        """
        Encode news titles using BERT
        Args:
            input_ids: tensor of shape [batch_size, seq_length] or [batch_size, history_length, seq_length]
            attention_mask: tensor of shape [batch_size, seq_length] or [batch_size, history_length, seq_length]
        Returns:
            tensor of shape [batch_size, hidden_size] or [batch_size, history_length, hidden_size]
        """
        # Check if we have hierarchical input (for user history)
        if len(input_ids.shape) == 3:  # [batch_size, history_length, seq_length]
            batch_size, history_length, seq_length = input_ids.shape
            
            # Reshape to [batch_size * history_length, seq_length]
            flat_input_ids = input_ids.view(-1, seq_length)
            flat_attention_mask = attention_mask.view(-1, seq_length)
            
            # Process with BERT
            outputs = self.bert(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                return_dict=True
            )
            
            # Use [CLS] token embedding as the news representation
            flat_news_vector = outputs.last_hidden_state[:, 0, :]
            flat_news_vector = self.dropout(flat_news_vector)
            
            # Reshape back to [batch_size, history_length, hidden_size]
            news_vector = flat_news_vector.view(batch_size, history_length, -1)
            
        else:  # Regular input [batch_size, seq_length]
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Use [CLS] token embedding as the news representation
            news_vector = outputs.last_hidden_state[:, 0, :]
            news_vector = self.dropout(news_vector)
        
        return news_vector
    
    def get_news_vector(self, news_features):
        """
        Get news vectors for a batch of news
        Args:
            news_features: dict containing 'input_ids' and 'attention_mask'
                Could be single-level (for candidate news) or hierarchical (for user history)
        Returns:
            tensor of shape [batch_size, hidden_size] or [batch_size, history_length, hidden_size]
        """
        return self.forward(
            input_ids=news_features['input_ids'],
            attention_mask=news_features['attention_mask']
        )