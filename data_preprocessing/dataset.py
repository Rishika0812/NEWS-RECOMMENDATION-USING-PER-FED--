import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MINDDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        """
        Initialize MIND dataset
        Args:
            behaviors_path: path to behaviors.csv
            news_path: path to news.csv
        """
        try:
            self.behaviors_df = pd.read_csv(behaviors_path)
            print(f"Successfully loaded behaviors file with columns: {self.behaviors_df.columns.tolist()}")
        except Exception as e:
            raise Exception(f"Error loading behaviors file: {str(e)}")

        try:
            self.news_df = pd.read_csv(news_path)
            print(f"Successfully loaded news file with columns: {self.news_df.columns.tolist()}")
            if 'News ID' not in self.news_df.columns:
                raise KeyError("'News ID' column not found in news.csv")
            if 'User ID' not in self.behaviors_df.columns or 'History' not in self.behaviors_df.columns or 'Impressions' not in self.behaviors_df.columns:
                raise KeyError("Required columns not found in behaviors.csv")
            # Convert column names to match the actual file format
            self.behaviors_df.columns = self.behaviors_df.columns.str.lower()
            self.behaviors_df = self.behaviors_df.rename(columns={'user id': 'user_id'})
        except Exception as e:
            raise Exception(f"Error loading news file: {str(e)}")
        
        # Process news data
        self.news_id_dict = {}
        for idx, row in self.news_df.iterrows():
            try:
                self.news_id_dict[row['News ID']] = idx
            except KeyError as e:
                raise KeyError(f"Error accessing News ID at index {idx}: {str(e)}")
            except Exception as e:
                raise Exception(f"Unexpected error processing news data at index {idx}: {str(e)}")
            
    def __len__(self):
        return len(self.behaviors_df)
    
    def __getitem__(self, idx):
        behavior = self.behaviors_df.iloc[idx]
        
        # Get user history
        history = behavior['history'].split()
        history_idx = [self.news_id_dict[h] for h in history if h in self.news_id_dict]
        
        # Get clicked and non-clicked news
        impressions = behavior['impressions'].split()
        clicked = []
        non_clicked = []
        
        for imp in impressions:
            try:
                news_id, label = imp.split('-')
                if news_id in self.news_id_dict:
                    if int(label) == 1:
                        clicked.append(self.news_id_dict[news_id])
                    else:
                        non_clicked.append(self.news_id_dict[news_id])
            except ValueError as e:
                print(f"Warning: Skipping invalid impression format: {imp}")
        
        return {
            'user_id': behavior['user_id'],
            'history': history_idx,
            'clicked': clicked,
            'non_clicked': non_clicked
        }