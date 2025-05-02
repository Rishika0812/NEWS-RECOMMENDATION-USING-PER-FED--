import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_privacy_accuracy_tradeoff(epsilons, accuracies, title="Privacy-Accuracy Trade-off"):
    """
    Plot the trade-off between privacy (epsilon) and model accuracy.
    
    Args:
        epsilons (list): List of privacy budget values (ε)
        accuracies (list): List of corresponding accuracy values
        title (str): Title of the plot
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=accuracies,
        mode='lines+markers',
        name='Trade-off curve'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Privacy Budget (ε)",
        yaxis_title="Model Accuracy",
        showlegend=True
    )
    return fig

def plot_noise_impact(noise_multipliers, metrics, metric_name="Accuracy"):
    """
    Visualize the impact of different noise multipliers on model performance.
    
    Args:
        noise_multipliers (list): Different noise multiplier values
        metrics (list): Corresponding metric values
        metric_name (str): Name of the metric being plotted
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=noise_multipliers,
        y=metrics,
        mode='lines+markers',
        name=f'{metric_name} vs Noise'
    ))
    
    fig.update_layout(
        title=f"Impact of Noise on {metric_name}",
        xaxis_title="Noise Multiplier",
        yaxis_title=metric_name,
        showlegend=True
    )
    return fig

def plot_client_metrics(client_ids, client_metrics, metric_name="Loss"):
    """
    Plot metrics across different clients.
    
    Args:
        client_ids (list): List of client identifiers
        client_metrics (list): List of metric values for each client
        metric_name (str): Name of the metric being plotted
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=client_ids,
        y=client_metrics,
        name=metric_name
    ))
    
    fig.update_layout(
        title=f"{metric_name} Across Clients",
        xaxis_title="Client ID",
        yaxis_title=metric_name,
        showlegend=True
    )
    return fig

def plot_training_progress(rounds, metrics, metric_names):
    """
    Plot training progress over federated learning rounds.
    
    Args:
        rounds (list): List of round numbers
        metrics (dict): Dictionary containing lists of metrics
        metric_names (list): Names of metrics to plot
    """
    fig = go.Figure()
    
    for metric_name in metric_names:
        fig.add_trace(go.Scatter(
            x=rounds,
            y=metrics[metric_name],
            mode='lines+markers',
            name=metric_name
        ))
    
    fig.update_layout(
        title="Training Progress Over Rounds",
        xaxis_title="Round Number",
        yaxis_title="Metric Value",
        showlegend=True
    )
    return fig

def create_interactive_privacy_dashboard(epsilon_range, noise_range, accuracy_data):
    """
    Create an interactive dashboard for privacy parameters.
    
    Args:
        epsilon_range (list): Range of epsilon values
        noise_range (list): Range of noise multiplier values
        accuracy_data (np.array): 2D array of accuracy values
    """
    fig = go.Figure(data=[
        go.Surface(z=accuracy_data, x=epsilon_range, y=noise_range)
    ])
    
    fig.update_layout(
        title='Privacy Parameters vs Accuracy',
        scene=dict(
            xaxis_title='Privacy Budget (ε)',
            yaxis_title='Noise Multiplier',
            zaxis_title='Accuracy'
        )
    )
    return fig