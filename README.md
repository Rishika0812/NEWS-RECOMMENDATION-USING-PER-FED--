# Privacy-Preserving Federated News Recommendation

A PhD-level project implementing a privacy-preserving federated learning system for personalized news recommendation using the MIND dataset.

## 🌟 Features

- **Federated Learning**: Train across multiple clients without sharing raw user data
- **Transformer-Based Recommendation**: Using BERT embeddings and NRMS architecture
- **Privacy-Preserving Techniques**:
  - Differential Privacy (DP-SGD)
  - Secure Aggregation
  - Homomorphic Encryption
- **Interactive Dashboard**: Visualize privacy-accuracy trade-offs and personalized recommendations

## 📋 Project Structure

```
project/
├── data/                     # MIND dataset files
│   ├── behaviors.csv         # User behaviors
│   └── news.csv              # News articles
├── data_preprocessing/       # Data preparation modules
│   ├── dataset.py            # MINDDataset implementation
│   ├── preprocessor.py       # News data preprocessing
│   └── tokenizer.py          # Text tokenization using BERT
├── models/                   # Neural network components
│   ├── bert_encoder.py       # BERT-based news encoder
│   ├── user_encoder.py       # User representation model
│   └── nrms.py               # NRMS recommendation architecture
├── federated/                # Federated learning components
│   ├── client.py             # Client-side training
│   ├── server.py             # Server aggregation
│   └── fedavg.py             # Federated Averaging implementation
├── privacy/                  # Privacy-preserving mechanisms
│   ├── dp_sgd.py             # Differentially Private SGD
│   ├── secure_aggregation.py # Secure aggregation protocols
│   └── homomorphic.py        # Homomorphic encryption utilities
├── evaluation/               # Evaluation metrics
│   ├── metrics.py            # Recommendation metrics (MRR, NDCG, etc.)
│   └── privacy_metrics.py    # Privacy budget analysis
├── visualisation/            # Interactive dashboard
│   ├── dashboard.py          # Streamlit dashboard implementation
│   └── visualization_utils.py # Plotting utilities
├── results/                  # Experimental results
│   └── experiment_results.csv # Training metrics data
├── main.py                   # Main training script
└── requirements.txt          # Project dependencies
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/privacy-preserving-news-recommendation.git
cd privacy-preserving-news-recommendation
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the MIND dataset:
```bash
# MIND Small dataset
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d data/
unzip MINDsmall_dev.zip -d data/
```

### Running the Project

1. Train the federated learning model:
```bash
python main.py
```

2. Start the interactive dashboard:
```bash
cd visualization
streamlit run dashboard.py
```

## 📊 System Components

### 1. Data Preprocessing

- `MINDDataset`: Processes and loads the MIND dataset
- `NewsPreprocessor`: Extracts and preprocesses news articles
- `NewsTokenizer`: Tokenizes news titles for BERT processing

### 2. Model Architecture

- `BertEncoder`: Encodes news titles using pre-trained BERT
- `UserEncoder`: Creates user representations from news history
- `NRMS`: Neural News Recommendation Model with Self-Attention

### 3. Federated Learning

- `FederatedClient`: Trains local models on user data
- `FederatedServer`: Aggregates client models securely
- `FedAvg`: Implements Federated Averaging algorithm

### 4. Privacy-Preserving Techniques

- `DPSGD`: Implements differentially private stochastic gradient descent
- `SecureAggregation`: Securely combines model updates without revealing individual updates
- `HomomorphicEncryption`: Enables computation on encrypted model parameters

### 5. Visualization

- Interactive dashboard showing:
  - Privacy-accuracy trade-offs
  - Client performance metrics
  - Training progress
  - Personalized news recommendations
  - Privacy budget analysis

## 🔒 Privacy Guarantees

Our system provides three layers of privacy protection:

1. **Federated Learning**: Data never leaves user devices
2. **Differential Privacy**: Protects against membership inference attacks
3. **Secure Aggregation**: Prevents leakage during model aggregation

The privacy budget (ε) is tracked throughout training to ensure compliance with privacy requirements.

## 📈 Performance Metrics

- **Recommendation Quality**: MRR, NDCG@5, NDCG@10, AUC
- **Privacy Cost**: ε-value (privacy budget), membership advantage
- **Computational Overhead**: Additional time due to privacy mechanisms

## 🧪 Experimental Results

Privacy-accuracy trade-off is visualized in the dashboard, showing that:
- Low privacy budget (ε < 1) significantly impacts model accuracy
- Medium privacy budget (1 < ε < 5) achieves good balance
- High privacy budget (ε > 5) provides minimal privacy protection

## 📚 References

1. Wu, F., et al. (2019). Neural News Recommendation with Multi-Head Self-Attention. EMNLP 2019.
2. McMahan, H. B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017.
3. Abadi, M., et al. (2016). Deep Learning with Differential Privacy. CCS 2016.
4. Bonawitz, K., et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. CCS 2017.

