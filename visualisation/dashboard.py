import streamlit as st
import numpy as np
import torch
import pandas as pd
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization_utils import (
    plot_privacy_accuracy_tradeoff,
    plot_noise_impact,
    plot_client_metrics,
    plot_training_progress,
    create_interactive_privacy_dashboard
)

from models.nrms import NRMS
from models.bert_encoder import BertEncoder
from models.user_encoder import UserEncoder
from data_preprocessing.dataset import MINDDataset
from data_preprocessing.preprocessor import NewsPreprocessor
from data_preprocessing.tokenizer import NewsTokenizer
from federated.client import FederatedClient
from federated.server import FederatedServer
from privacy.dp_sgd import DPSGD
from evaluation.privacy_metrics import compute_epsilon, estimate_privacy_risk

# Load global experiment results if available
def load_results():
    try:
        # Try to load saved results from previous runs
        return pd.read_csv('results/experiment_results.csv')
    except:
        # If file doesn't exist, return None
        st.warning("No experiment results found. Run training first or use simulated data.")
        return None

# Get real news recommendations 
def get_news_recommendations(model, user_id, news_df, tokenizer, top_k=5, personalization_params=None):
    try:
        # Default personalization parameters if none provided
        if personalization_params is None:
            personalization_params = {
                "enabled": True,
                "strength": 0.7,
                "history_weight": 0.8,
                "diversity": 0.5
            }
            
        # Check which column to use for titles (accommodate different column name formats)
        title_column = None
        for possible_title in ['title', 'Title', 'TITLE', 'news_title']:
            if possible_title in news_df.columns:
                title_column = possible_title
                break
        
        if title_column is None:
            # If we can't find a title column, let's make a best guess based on the columns we have
            if len(news_df.columns) > 1:  # Use the second column as a fallback
                title_column = news_df.columns[1]
            else:
                raise ValueError("Could not identify a title column in the news dataframe")
                
        news_titles = news_df[title_column].values
        
        # Fix tokenization - ensure we're passing a list of strings to the tokenizer
        # Convert numpy array to list of strings
        news_titles_list = [str(title) for title in news_titles]
        
        # Process titles one by one to avoid batch processing issues
        news_embeddings_list = []
        for title in news_titles_list:
            # Tokenize a single title at a time
            title_features = tokenizer.tokenize(title)
            # Get embedding for this title
            title_embedding = model.get_news_embedding(title_features)
            news_embeddings_list.append(title_embedding)
        
        # Stack all embeddings
        news_embeddings = torch.cat(news_embeddings_list, dim=0)
        
        # Generate user embedding
        if personalization_params["enabled"]:
            # This is a simplified version - in reality we'd need to encode the user history properly
            # For demo purposes, we're combining a random user history with personalization settings
            # Simulate a personalized user embedding
            personal_embedding = torch.randn(1, 768)  # Personal interests
            general_embedding = torch.ones(1, 768) / 768**0.5  # General popular interests
            
            # Apply personalization strength
            p_strength = personalization_params["strength"]
            user_embedding = p_strength * personal_embedding + (1 - p_strength) * general_embedding
            user_embedding = user_embedding / user_embedding.norm()
        else:
            # If personalization is disabled, use general popularity embedding
            user_embedding = torch.ones(1, 768) / 768**0.5
        
        # Calculate relevance scores (cosine similarity)
        scores = torch.matmul(user_embedding, news_embeddings.transpose(0, 1)).squeeze()
        
        # Apply diversity factor if personalization is enabled
        if personalization_params["enabled"] and personalization_params["diversity"] > 0:
            # Simple diversity implementation - add small random noise to scores
            # Higher diversity_factor means more random variation
            diversity_noise = torch.randn_like(scores) * personalization_params["diversity"] * 0.1
            scores = scores + diversity_noise
        
        # Get top-k news
        top_indices = torch.topk(scores, min(top_k, len(news_df))).indices.numpy()
        
        recommendations = []
        for idx in top_indices:
            # Also be flexible with abstract/content column
            content_column = None
            for possible_content in ['abstract', 'Abstract', 'content', 'Content', 'body', 'Body']:
                if possible_content in news_df.columns:
                    content_column = possible_content
                    break
            
            content = "No content available"
            if content_column is not None:
                content = news_df.iloc[idx].get(content_column, "No content available")
                
            recommendations.append({
                "title": news_df.iloc[idx][title_column],
                "score": scores[idx].item(),
                "content": content
            })
            
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        # Fallback to example data
        return [
            {"title": "Latest Developments in AI", "score": 0.95, "content": "AI continues to advance rapidly..."},
            {"title": "Global Economic Trends", "score": 0.85, "content": "Economic indicators suggest..."},
            {"title": "Breakthrough in Quantum Computing", "score": 0.82, "content": "Researchers have achieved..."},
            {"title": "Climate Change Impact Study", "score": 0.78, "content": "New research shows..."},
            {"title": "Space Exploration Updates", "score": 0.75, "content": "NASA's latest mission..."}
        ]

def main():
    st.set_page_config(page_title="Federated News Recommendation Dashboard", layout="wide")
    st.title("📰 Privacy-Preserving Federated News Recommendation")
    
    # Load experiment results
    results_df = load_results()
    
    # Sidebar for privacy parameters
    st.sidebar.title("Privacy Settings")
    noise_multiplier = st.sidebar.slider(
        "Noise Multiplier",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls the amount of noise added for differential privacy"
    )
    
    clipping_threshold = st.sidebar.slider(
        "Gradient Clipping Threshold",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Maximum L2 norm of gradient updates"
    )
    
    # Calculate privacy budget for the chosen parameters
    iterations = 100  # Assuming 100 training iterations as an example
    sample_rate = 0.01  # Assuming 1% sampling rate as an example
    delta = 1e-5
    
    epsilon = compute_epsilon(noise_multiplier, sample_rate, iterations, delta)
    privacy_risk = estimate_privacy_risk(epsilon, delta)
    
    st.sidebar.metric("Privacy Budget (ε)", f"{epsilon:.2f}")
    st.sidebar.metric("Privacy Risk", f"{privacy_risk:.4f}")
    
    # Add button to run training with these parameters (not functional in this demo)
    if st.sidebar.button("Run Training with These Parameters"):
        st.sidebar.info("Training functionality would be triggered here with the selected parameters")
    
    # Main dashboard content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Privacy-Accuracy Trade-off")
        if results_df is not None:
            # Use real data if available
            epsilons = results_df['epsilon'].values
            accuracies = results_df['accuracy'].values
        else:
            # Use simulated data as fallback
            epsilons = np.linspace(0.1, 10, 20)
            accuracies = 0.9 - 0.1 * np.exp(-0.3 * epsilons)
            
        fig = plot_privacy_accuracy_tradeoff(epsilons, accuracies)
        st.plotly_chart(fig)
        
        st.subheader("Client Performance Metrics")
        if results_df is not None and 'client_id' in results_df.columns:
            # Use real client data if available
            client_data = results_df.groupby('client_id')['accuracy'].mean().reset_index()
            client_ids = [f"Client {i}" for i in client_data['client_id']]
            client_metrics = client_data['accuracy'].values
        else:
            # Use simulated data as fallback
            client_ids = [f"Client {i}" for i in range(1, 11)]
            client_metrics = np.random.uniform(0.7, 0.9, 10)
            
        fig = plot_client_metrics(client_ids, client_metrics, "Accuracy")
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Impact of Noise on Model Performance")
        if results_df is not None and 'noise_multiplier' in results_df.columns:
            # Use real noise impact data if available
            noise_data = results_df.groupby('noise_multiplier')['accuracy'].mean().reset_index()
            noise_range = noise_data['noise_multiplier'].values
            metrics = noise_data['accuracy'].values
        else:
            # Use simulated data as fallback
            noise_range = np.linspace(0.1, 2.0, 20)
            metrics = 0.9 - 0.15 * np.exp(noise_range)
            
        fig = plot_noise_impact(noise_range, metrics)
        st.plotly_chart(fig)
        
        st.subheader("Training Progress")
        if results_df is not None and 'round' in results_df.columns:
            # Use real training progress data if available
            rounds_data = results_df.groupby('round')[['accuracy', 'loss']].mean().reset_index()
            rounds = rounds_data['round'].values
            metrics = {
                "Accuracy": rounds_data['accuracy'].values,
                "Loss": rounds_data['loss'].values
            }
        else:
            # Use simulated data as fallback
            rounds = list(range(1, 21))
            metrics = {
                "Accuracy": 0.5 + 0.4 * (1 - np.exp(-0.2 * np.array(rounds))),
                "Loss": 1.0 * np.exp(-0.15 * np.array(rounds))
            }
            
        fig = plot_training_progress(rounds, metrics, ["Accuracy", "Loss"])
        st.plotly_chart(fig)
    
    # News Recommendations Section
    st.subheader("📱 Personalized News Recommendations")
    
    # Add personalization controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Personalization Controls")
    
    enable_personalization = st.sidebar.toggle(
        "Enable Personalization",
        value=True,
        help="Toggle personalization on/off. When off, recommendations will be based only on general popularity."
    )
    
    if enable_personalization:
        personalization_strength = st.sidebar.slider(
            "Personalization Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls how much recommendations are tailored to user preferences vs general popularity"
        )
        
        use_history_weight = st.sidebar.slider(
            "User History Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Weight given to user's reading history for recommendations"
        )
        
        diversity_factor = st.sidebar.slider(
            "Content Diversity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values increase topic diversity in recommendations"
        )
    
    # Initialize model for recommendations
    try:
        # Initialize model components
        news_encoder = BertEncoder()
        user_encoder = UserEncoder()
        model = NRMS(news_encoder, user_encoder)
        
        # Try to load the latest model checkpoint
        try:
            model.load_state_dict(torch.load('models/latest_model.pt'))
            model.eval()
            st.success("Loaded trained model for recommendations")
        except:
            st.warning("No trained model found, using untrained model for demo purposes")
        
        # Try to load news data
        try:
            # Try multiple possible paths for data files
            possible_paths = [
                # From project root
                {'behaviors_path': os.path.join('data', 'behaviors.csv'), 
                 'news_path': os.path.join('data', 'news.csv')},
                # From one level up (relative to visualisation folder)
                {'behaviors_path': os.path.join('..', 'data', 'behaviors.csv'), 
                 'news_path': os.path.join('..', 'data', 'news.csv')},
                # Absolute path if available
                {'behaviors_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'behaviors.csv')),
                 'news_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'news.csv'))}
            ]
            
            dataset = None
            for paths in possible_paths:
                try:
                    if os.path.exists(paths['behaviors_path']) and os.path.exists(paths['news_path']):
                        dataset = MINDDataset(
                            behaviors_path=paths['behaviors_path'],
                            news_path=paths['news_path']
                        )
                        st.success(f"Successfully loaded data from {paths['behaviors_path']}")
                        break
                except Exception as e:
                    st.warning(f"Failed to load data from {paths['behaviors_path']}: {e}")
                    continue
            
            if dataset is None:
                raise FileNotFoundError("Could not find behaviors.csv and news.csv files in any of the attempted paths")
                
            news_df = dataset.news_df
            tokenizer = NewsTokenizer()
            
            # Let user select a user ID
            user_ids = dataset.behaviors_df['user_id'].unique()
            selected_user = st.selectbox("Select User ID for Recommendations", 
                                        options=user_ids if len(user_ids) > 0 else ["user_1", "user_2", "user_3"])
            
            # Get recommendations for the selected user
            personalization_params = {
                "enabled": enable_personalization,
                "strength": personalization_strength if enable_personalization else 0.0,
                "history_weight": use_history_weight if enable_personalization else 0.0,
                "diversity": diversity_factor if enable_personalization else 0.5
            }
            
            # Add this button to refresh recommendations with current settings
            if st.button("Generate Personalized Recommendations", type="primary"):
                st.success("Generating recommendations with personalization settings: " + 
                          f"Enabled: {personalization_params['enabled']}, " +
                          f"Strength: {personalization_params['strength']:.1f}, " +
                          f"History Weight: {personalization_params['history_weight']:.1f}, " +
                          f"Diversity: {personalization_params['diversity']:.1f}")
                
                # Get recommendations with personalization parameters
                recommendations = get_news_recommendations(
                    model, 
                    selected_user, 
                    news_df, 
                    tokenizer,
                    personalization_params=personalization_params
                )
            else:
                # Get recommendations with personalization parameters
                recommendations = get_news_recommendations(
                    model, 
                    selected_user, 
                    news_df, 
                    tokenizer,
                    personalization_params=personalization_params
                )
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            # Fallback to example news
            recommendations = [
                {"title": "Latest Developments in AI", "score": 0.95, "content": "AI continues to advance rapidly..."},
                {"title": "Global Economic Trends", "score": 0.85, "content": "Economic indicators suggest..."},
                {"title": "Breakthrough in Quantum Computing", "score": 0.82, "content": "Researchers have achieved..."},
                {"title": "Climate Change Impact Study", "score": 0.78, "content": "New research shows..."},
                {"title": "Space Exploration Updates", "score": 0.75, "content": "Space agencies announce new missions..."}
            ]
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        # Fallback to example news
        recommendations = [
            {"title": "Latest Developments in AI", "score": 0.95, "content": "AI continues to advance rapidly..."},
            {"title": "Global Economic Trends", "score": 0.85, "content": "Economic indicators suggest..."},
            {"title": "Breakthrough in Quantum Computing", "score": 0.82, "content": "Researchers have achieved..."},
            {"title": "Climate Change Impact Study", "score": 0.78, "content": "New research shows..."},
            {"title": "Space Exploration Updates", "score": 0.75, "content": "Space agencies announce new missions..."}
        ]
    
    # Display recommendations
    st.subheader("Your Recommended Articles")
    for news in recommendations:
        with st.expander(f"{news['title']} (Relevance: {news['score']:.2f})"):
            st.write(news['content'])
    
    # Privacy Analysis Section
    st.subheader("🔒 Privacy Analysis")
    
    # Create a 3D privacy dashboard showing the relationship between noise, epsilon, and accuracy
    epsilon_range = np.linspace(0.1, 5, 20)
    noise_range = np.linspace(0.1, 2, 20)
    
    if results_df is not None and 'epsilon' in results_df.columns and 'noise_multiplier' in results_df.columns:
        # Try to use real data for the 3D plot
        try:
            # Create a grid for the heatmap
            X, Y = np.meshgrid(epsilon_range, noise_range)
            
            # Interpolate accuracy values for the grid points
            from scipy.interpolate import griddata
            points = results_df[['epsilon', 'noise_multiplier']].values
            values = results_df['accuracy'].values
            Z = griddata(points, values, (X, Y), method='cubic', fill_value=0.7)
        except:
            # Fallback to simulated data
            X, Y = np.meshgrid(epsilon_range, noise_range)
            Z = 0.9 - 0.2 * np.exp(-0.3 * X) - 0.1 * Y
    else:
        # Use simulated data
        X, Y = np.meshgrid(epsilon_range, noise_range)
        Z = 0.9 - 0.2 * np.exp(-0.3 * X) - 0.1 * Y
    
    fig = create_interactive_privacy_dashboard(epsilon_range, noise_range, Z)
    st.plotly_chart(fig)
    
    # Help section
    with st.expander("ℹ️ About the Dashboard"):
        st.markdown("""
        This dashboard visualizes the privacy-accuracy trade-off in federated learning for news recommendation.
        
        - **Privacy Settings**: Adjust the noise level and clipping threshold for differential privacy
        - **Privacy-Accuracy Trade-off**: Shows how accuracy varies with privacy budget (epsilon)
        - **Client Performance**: Compares performance across different clients
        - **Noise Impact**: Shows how model performance changes with different noise multipliers
        - **Training Progress**: Tracks accuracy and loss over training rounds
        - **Personalized News**: Shows recommended news articles for the selected user
        - **Privacy Analysis**: 3D visualization of the relationship between privacy parameters and accuracy
        
        The system uses BERT embeddings for news representation and implements privacy-preserving federated learning
        with differential privacy and secure aggregation.
        """)

if __name__ == "__main__":
    main()