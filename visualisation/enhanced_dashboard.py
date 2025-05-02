import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
try:
    from evaluation.privacy_metrics import compute_epsilon, estimate_privacy_risk
except ImportError:
    # Fallback implementations if modules are not available
    def compute_epsilon(noise_multiplier, sample_rate, iterations, delta):
        return np.sqrt(2 * np.log(1.25 / delta)) * noise_multiplier * np.sqrt(iterations * sample_rate)
    
    def estimate_privacy_risk(epsilon, delta):
        return 1 - np.exp(-epsilon)

# Configure page
st.set_page_config(
    page_title="Privacy-Preserving Federated News Recommender",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .header-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
    }
    .subheader-text {
        font-size: 1.8rem;
        color: #1E3A8A;
        padding: 0.5rem 0;
    }
    .metric-container {
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #1E3A8A;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2563EB;
    }
    .info-container {
        background-color: #DBEAFE;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .warning-container {
        background-color: #FEF3C7;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load experiment results
@st.cache_data
def load_results():
    try:
        if os.path.exists('results/experiment_results.csv'):
            df = pd.read_csv('results/experiment_results.csv')
            return df
        else:
            st.warning("No experiment results found. Using simulated data.")
            return None
    except Exception as e:
        st.error(f"Error loading experiment results: {str(e)}")
        return None

def simulate_federated_training_data():
    """Generate simulated data for demonstration when real data is not available"""
    # Generate rounds from 1 to 10
    rounds = np.arange(1, 11)
    
    # Client IDs from 0 to 4
    client_ids = np.arange(5)
    
    # Noise multiplier values from 0.1 to 2.0
    noise_multipliers = np.linspace(0.1, 2.0, 10)
    
    # Generate data points
    data = []
    
    for round_idx in rounds:
        epsilon_base = 0.1 * round_idx  # Epsilon increases with rounds
        
        for client_id in client_ids:
            # For each client, use different noise multipliers in different rounds
            noise_idx = (round_idx + client_id) % len(noise_multipliers)
            noise_multiplier = noise_multipliers[noise_idx]
            
            # Compute epsilon based on noise
            epsilon = epsilon_base * (1.0 / noise_multiplier)
            
            # Base accuracy that improves with rounds but decreases with more noise
            base_accuracy = 0.5 + (0.04 * round_idx) - (0.05 * noise_multiplier)
            # Add client-specific variation
            accuracy = base_accuracy + 0.02 * client_id
            # Ensure accuracy is within [0, 1]
            accuracy = max(0, min(1, accuracy))
            
            # Loss is roughly inverse of accuracy
            loss = 1.0 - (0.7 * accuracy)
            
            data.append({
                'round': round_idx,
                'client_id': client_id,
                'epsilon': epsilon,
                'noise_multiplier': noise_multiplier,
                'accuracy': accuracy,
                'loss': loss
            })
    
    return pd.DataFrame(data)

# Main application
def main():
    # Header
    st.markdown('<p class="header-text">🔒 Privacy-Preserving Federated News Recommendation System</p>', 
                unsafe_allow_html=True)
    
    # Information about the project
    with st.expander("ℹ️ About this system", expanded=False):
        st.markdown("""
        ### System Overview
        
        This dashboard visualizes a privacy-preserving federated news recommendation system with the following components:
        
        1. **News Recommendation Model (NRMS)**: Neural model with news and user encoders for personalized recommendations
        2. **Federated Learning**: Distributed training approach where models are trained on client devices
        3. **Differential Privacy**: Privacy-preserving technique (DP-SGD) that protects user data during training
        
        ### Privacy-Utility Trade-off
        
        The system demonstrates the balance between:
        - **Privacy**: Protected by differential privacy with noise multiplier and gradient clipping
        - **Utility**: Recommendation quality measured by accuracy, AUC, and ranking metrics
        
        Explore the dashboard to see how different privacy settings affect model performance.
        """)
    
    # Load results
    results_df = load_results()
    
    # If no real data, use simulated data
    if results_df is None:
        results_df = simulate_federated_training_data()
    
    # Sidebar for filtering options
    st.sidebar.markdown("## Dashboard Controls")
    
    # Filter by federated learning rounds
    available_rounds = sorted(results_df['round'].unique())
    selected_rounds = st.sidebar.multiselect(
        "Filter by Round",
        options=available_rounds,
        default=available_rounds
    )
    
    # Filter by clients
    available_clients = sorted(results_df['client_id'].unique())
    selected_clients = st.sidebar.multiselect(
        "Filter by Client",
        options=available_clients,
        default=available_clients
    )
    
    # Filter by noise multiplier
    min_noise = float(results_df['noise_multiplier'].min())
    max_noise = float(results_df['noise_multiplier'].max())
    noise_range = st.sidebar.slider(
        "Noise Multiplier Range",
        min_value=min_noise,
        max_value=max_noise,
        value=(min_noise, max_noise),
        step=0.1
    )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if selected_rounds:
        filtered_df = filtered_df[filtered_df['round'].isin(selected_rounds)]
    
    if selected_clients:
        filtered_df = filtered_df[filtered_df['client_id'].isin(selected_clients)]
    
    filtered_df = filtered_df[
        (filtered_df['noise_multiplier'] >= noise_range[0]) & 
        (filtered_df['noise_multiplier'] <= noise_range[1])
    ]
    
    # Add privacy simulation controls
    st.sidebar.markdown("## Privacy Simulation")
    
    simulation_noise = st.sidebar.slider(
        "Simulated Noise Multiplier",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    simulation_clip = st.sidebar.slider(
        "Simulated Gradient Clipping",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    
    # Calculate simulated privacy budget
    sample_rate = st.sidebar.slider(
        "Sample Rate",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f"
    )
    
    iterations = st.sidebar.slider(
        "Training Iterations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    delta = 1e-5
    simulated_epsilon = compute_epsilon(simulation_noise, sample_rate, iterations, delta)
    simulated_risk = estimate_privacy_risk(simulated_epsilon, delta)
    
    st.sidebar.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="metric-label">Simulated Privacy Budget (ε)</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="metric-value">{simulated_epsilon:.2f}</div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="metric-container" style="margin-top: 10px;">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="metric-label">Simulated Privacy Risk</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="metric-value">{simulated_risk:.4f}</div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main dashboard content
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filter settings.")
    else:
        # Key metrics section
        st.markdown('<p class="subheader-text">Key Performance Metrics</p>', unsafe_allow_html=True)
        
        # Get key metrics
        avg_accuracy = filtered_df['accuracy'].mean()
        avg_loss = filtered_df['loss'].mean()
        avg_epsilon = filtered_df['epsilon'].mean()
        best_client_id = filtered_df.groupby('client_id')['accuracy'].mean().idxmax()
        max_round = filtered_df['round'].max()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Average Accuracy</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_accuracy:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Average Loss</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_loss:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Privacy Budget (ε)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{avg_epsilon:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Best Performing Client</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">Client {best_client_id}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Privacy-Utility Tradeoff", 
            "Federated Learning Progress", 
            "Client Performance", 
            "Recommendation System"
        ])
        
        with tab1:
            st.markdown("### Privacy-Utility Tradeoff Analysis")
            
            st.markdown("""
            <div class="info-container">
            <strong>How to interpret:</strong> These charts show how different privacy settings (noise multiplier and privacy budget ε) 
            affect model utility (accuracy). Higher noise and lower epsilon provide stronger privacy guarantees but may reduce accuracy.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Privacy budget (epsilon) vs. accuracy
                epsilon_vs_acc = filtered_df.groupby('epsilon')['accuracy'].mean().reset_index()
                
                fig = px.scatter(
                    epsilon_vs_acc, 
                    x='epsilon', 
                    y='accuracy',
                    size=[10] * len(epsilon_vs_acc),
                    labels={"epsilon": "Privacy Budget (ε)", "accuracy": "Model Accuracy"},
                    title="Privacy Budget vs. Accuracy"
                )
                
                fig.update_layout(
                    xaxis_title="Privacy Budget (ε) - Lower is More Private",
                    yaxis_title="Accuracy",
                    showlegend=False
                )
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=epsilon_vs_acc['epsilon'],
                        y=epsilon_vs_acc['accuracy'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='rgba(31, 119, 180, 0.5)', width=2)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Noise multiplier vs. accuracy
                noise_vs_acc = filtered_df.groupby('noise_multiplier')['accuracy'].mean().reset_index()
                
                fig = px.scatter(
                    noise_vs_acc, 
                    x='noise_multiplier', 
                    y='accuracy',
                    size=[10] * len(noise_vs_acc),
                    labels={"noise_multiplier": "Noise Multiplier", "accuracy": "Model Accuracy"},
                    title="Noise Multiplier vs. Accuracy"
                )
                
                fig.update_layout(
                    xaxis_title="Noise Multiplier - Higher is More Private",
                    yaxis_title="Accuracy",
                    showlegend=False
                )
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=noise_vs_acc['noise_multiplier'],
                        y=noise_vs_acc['accuracy'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='rgba(255, 127, 14, 0.5)', width=2)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 3D visualization of privacy-utility tradeoff
            st.markdown("### 3D Privacy-Utility-Noise Relationship")
            
            # Prepare data for 3D visualization
            pivot_data = filtered_df.pivot_table(
                index='noise_multiplier', 
                columns='epsilon', 
                values='accuracy',
                aggfunc='mean'
            ).reset_index()
            
            # Convert to format suitable for 3D visualization
            x_vals = pivot_data.columns[1:].astype(float)  # epsilon values
            y_vals = pivot_data['noise_multiplier'].values  # noise multiplier values
            z_vals = pivot_data.iloc[:, 1:].values  # accuracy values
            
            fig = go.Figure(data=[go.Surface(z=z_vals, x=x_vals, y=y_vals)])
            
            fig.update_layout(
                title='3D Privacy-Utility Tradeoff',
                width=800,
                height=600,
                scene=dict(
                    xaxis_title='Privacy Budget (ε)',
                    yaxis_title='Noise Multiplier',
                    zaxis_title='Accuracy',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Federated Learning Progress")
            
            st.markdown("""
            <div class="info-container">
            <strong>How to interpret:</strong> These charts show the progression of model performance across federated learning rounds.
            Each round represents one cycle of the federated averaging algorithm where client models are aggregated.
            </div>
            """, unsafe_allow_html=True)
            
            # Get average metrics per round
            round_metrics = filtered_df.groupby('round').agg({
                'accuracy': 'mean',
                'loss': 'mean',
                'epsilon': 'mean',
                'noise_multiplier': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy progress
                fig = px.line(
                    round_metrics,
                    x='round',
                    y='accuracy',
                    markers=True,
                    labels={"round": "Training Round", "accuracy": "Average Accuracy"},
                    title="Accuracy Progress Over Federated Learning Rounds"
                )
                
                fig.update_layout(
                    xaxis_title="Training Round",
                    yaxis_title="Average Accuracy",
                    showlegend=False
                )
                
                # Add points with noise multiplier as hover info
                fig.add_trace(
                    go.Scatter(
                        x=round_metrics['round'],
                        y=round_metrics['accuracy'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=round_metrics['noise_multiplier'],
                            colorbar=dict(title="Noise"),
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[f"Noise: {noise:.2f}<br>ε: {eps:.2f}" 
                              for noise, eps in zip(round_metrics['noise_multiplier'], round_metrics['epsilon'])],
                        hoverinfo='text+x+y',
                        name='Rounds'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Loss progress
                fig = px.line(
                    round_metrics,
                    x='round',
                    y='loss',
                    markers=True,
                    labels={"round": "Training Round", "loss": "Average Loss"},
                    title="Loss Progress Over Federated Learning Rounds"
                )
                
                fig.update_layout(
                    xaxis_title="Training Round",
                    yaxis_title="Average Loss",
                    showlegend=False
                )
                
                # Add points with noise multiplier as hover info
                fig.add_trace(
                    go.Scatter(
                        x=round_metrics['round'],
                        y=round_metrics['loss'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=round_metrics['noise_multiplier'],
                            colorbar=dict(title="Noise"),
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[f"Noise: {noise:.2f}<br>ε: {eps:.2f}" 
                              for noise, eps in zip(round_metrics['noise_multiplier'], round_metrics['epsilon'])],
                        hoverinfo='text+x+y',
                        name='Rounds'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Privacy budget growth
            st.markdown("### Privacy Budget Accumulation")
            
            # Plot privacy budget growth across rounds
            fig = px.line(
                round_metrics,
                x='round',
                y='epsilon',
                markers=True,
                labels={"round": "Training Round", "epsilon": "Privacy Budget (ε)"},
                title="Privacy Budget Accumulation Over Training Rounds"
            )
            
            fig.update_layout(
                xaxis_title="Training Round",
                yaxis_title="Privacy Budget (ε)",
                showlegend=False,
                annotations=[dict(
                    x=0.5,
                    y=1.05,
                    showarrow=False,
                    text="Privacy budget increases with training duration",
                    xref="paper",
                    yref="paper",
                    font=dict(size=14)
                )]
            )
            
            # Add reference line for typical privacy budget threshold
            fig.add_shape(
                type="line",
                x0=min(round_metrics['round']),
                y0=1.0,
                x1=max(round_metrics['round']),
                y1=1.0,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
            )
            
            fig.add_annotation(
                x=max(round_metrics['round']),
                y=1.0,
                text="Typical privacy threshold (ε=1.0)",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Client Performance Analysis")
            
            st.markdown("""
            <div class="info-container">
            <strong>How to interpret:</strong> These visualizations show performance variations across different clients in the federation.
            Variations can be due to different data distributions, client capabilities, or privacy settings.
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate client-level metrics
            client_metrics = filtered_df.groupby('client_id').agg({
                'accuracy': ['mean', 'std', 'min', 'max'],
                'loss': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            # Flatten multi-level column index
            client_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in client_metrics.columns.values]
            
            # Select a specific round for detailed comparison
            round_selection = st.selectbox(
                "Select round for detailed client comparison:",
                options=sorted(filtered_df['round'].unique())
            )
            
            round_data = filtered_df[filtered_df['round'] == round_selection]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Client accuracy comparison for selected round
                fig = px.bar(
                    round_data,
                    x='client_id',
                    y='accuracy',
                    labels={"client_id": "Client ID", "accuracy": "Accuracy"},
                    title=f"Client Accuracy Comparison (Round {round_selection})",
                    color='accuracy',
                    color_continuous_scale='Blues'
                )
                
                fig.update_layout(
                    xaxis_title="Client ID",
                    yaxis_title="Accuracy",
                    xaxis=dict(type='category')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Client loss comparison for selected round
                fig = px.bar(
                    round_data,
                    x='client_id',
                    y='loss',
                    labels={"client_id": "Client ID", "loss": "Loss"},
                    title=f"Client Loss Comparison (Round {round_selection})",
                    color='loss',
                    color_continuous_scale='Reds_r'  # Reversed so lower loss is blue (better)
                )
                
                fig.update_layout(
                    xaxis_title="Client ID",
                    yaxis_title="Loss",
                    xaxis=dict(type='category')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Client performance across all rounds
            st.markdown("### Client Performance Across All Rounds")
            
            # Create data for heatmap
            accuracy_pivot = filtered_df.pivot_table(
                index='client_id',
                columns='round',
                values='accuracy',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                accuracy_pivot,
                labels=dict(x="Round", y="Client ID", color="Accuracy"),
                x=accuracy_pivot.columns,
                y=accuracy_pivot.index,
                color_continuous_scale='Blues',
                title="Client Accuracy Heatmap Across Rounds"
            )
            
            fig.update_layout(
                xaxis_title="Training Round",
                yaxis_title="Client ID",
                coloraxis_colorbar=dict(
                    title="Accuracy",
                    titleside="right"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Privacy impact on different clients
            st.markdown("### Impact of Privacy on Client Performance")
            
            # Get client data for different noise levels
            client_noise_acc = filtered_df.groupby(['client_id', 'noise_multiplier'])['accuracy'].mean().reset_index()
            
            fig = px.line(
                client_noise_acc,
                x='noise_multiplier',
                y='accuracy',
                color='client_id',
                markers=True,
                labels={"noise_multiplier": "Noise Multiplier", "accuracy": "Accuracy", "client_id": "Client ID"},
                title="Impact of Noise on Client Accuracy"
            )
            
            fig.update_layout(
                xaxis_title="Noise Multiplier (higher = more privacy)",
                yaxis_title="Accuracy",
                legend_title="Client ID"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### News Recommendation System Analysis")
            
            st.markdown("""
            <div class="info-container">
            <strong>About the recommendation model:</strong> The system uses a Neural Recommendation Model with Attention (NRMS) 
            trained via federated learning with differential privacy. This approach keeps user data on their devices while 
            protecting privacy through noise addition.
            </div>
            """, unsafe_allow_html=True)
            
            # Simulated recommendation examples
            st.markdown("#### Sample News Recommendations")
            
            # Generate some example news recommendations
            example_news = [
                {"title": "Latest Developments in AI", "score": 0.85 + 0.1 * np.random.random(), "category": "Technology"},
                {"title": "Economic Forecast for Q3 2023", "score": 0.80 + 0.1 * np.random.random(), "category": "Business"},
                {"title": "Breakthrough in Quantum Computing", "score": 0.75 + 0.1 * np.random.random(), "category": "Science"},
                {"title": "New Environmental Policy Announced", "score": 0.70 + 0.1 * np.random.random(), "category": "Politics"},
                {"title": "Major Sports Tournament Results", "score": 0.65 + 0.1 * np.random.random(), "category": "Sports"}
            ]
            
            example_df = pd.DataFrame(example_news)
            
            fig = px.bar(
                example_df,
                x='title',
                y='score',
                color='category',
                labels={"title": "News Article", "score": "Recommendation Score", "category": "Category"},
                title="Sample Personalized News Recommendations"
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Recommendation Score",
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Privacy impact on recommendation quality
            st.markdown("#### Privacy Impact on Recommendation Quality")
            
            # Create simulated data for recommendation quality vs privacy
            privacy_rec_data = pd.DataFrame({
                'epsilon': np.linspace(0.1, 3.0, 10),
                'recommendation_quality': 0.4 + 0.5 * (1 - np.exp(-1.2 * np.linspace(0.1, 3.0, 10))),
                'user_privacy': 1.0 - 0.6 * (1 - np.exp(-0.8 * np.linspace(0.1, 3.0, 10)))
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=privacy_rec_data['epsilon'],
                y=privacy_rec_data['recommendation_quality'],
                mode='lines+markers',
                name='Recommendation Quality',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=privacy_rec_data['epsilon'],
                y=privacy_rec_data['user_privacy'],
                mode='lines+markers',
                name='User Privacy',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title='Privacy-Utility Tradeoff in Recommendations',
                xaxis_title='Privacy Budget (ε)',
                yaxis_title='Score',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Add annotation for optimal point
            optimal_idx = 5  # Example optimal point
            fig.add_annotation(
                x=privacy_rec_data['epsilon'][optimal_idx],
                y=(privacy_rec_data['recommendation_quality'][optimal_idx] + 
                   privacy_rec_data['user_privacy'][optimal_idx]) / 2,
                text="Optimal Trade-off Point",
                showarrow=True,
                arrowhead=3,
                ax=40,
                ay=-40
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation of the recommendation process
            st.markdown("#### How Privacy-Preserving Recommendations Work")
            
            st.markdown("""
            1. **User History Encoding**: Convert user reading history into embeddings
            2. **News Content Encoding**: Encode news articles into embeddings
            3. **Federated Learning**: Train the model across clients without sharing raw data
            4. **Differential Privacy**: Add calibrated noise to protect individual privacy
            5. **Personalized Recommendations**: Generate recommendations based on user-news similarity
            
            The system balances recommendation quality with user privacy through DP-SGD, a differentially private training method.
            """)
    
    # Show raw data if requested
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)
        
        # Add download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="privacy_federated_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main() 