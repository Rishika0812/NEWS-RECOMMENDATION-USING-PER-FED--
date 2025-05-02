from transformers import BertTokenizer
import torch

class NewsTokenizer:
    def __init__(self, model_name='bert-base-uncased', max_length=30):
        """
        Initialize the news tokenizer
        Args:
            model_name: name of the pre-trained model to use
            max_length: maximum sequence length
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def tokenize(self, texts):
        """
        Tokenize a list of texts
        Args:
            texts: list of text strings to tokenize
        Returns:
            dict containing input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def batch_tokenize(self, text_batch, device='cpu'):
        """
        Tokenize a batch of texts and move to specified device
        Args:
            text_batch: list of text strings
            device: device to move tensors to
        Returns:
            dict containing input_ids and attention_mask
        """
        encoded = self.tokenize(text_batch)
        return {
            'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device)
        }
