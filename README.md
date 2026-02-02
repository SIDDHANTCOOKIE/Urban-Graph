# 🏙️ Urban-Graph: Network Resilience Analyzer

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.7-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

Spatio-Temporal GNNs for predicting urban network resilience using **Pre-Trained** Gated Graph Convolutions.

## 🎯 Goal
Model urban traffic as a dynamic graph to predict network disruption propagation using Gated Graph Convolutions (GatedGCN). Showcases mastery of Geometric Deep Learning and non-Euclidean data structures—the primary tech stack for Jet-as-Graph classification.

## 🛠️ Technologies
- **PyTorch Geometric (PyG)** - Graph Neural Networks library
- **NetworkX** - Graph manipulation and analysis
- **Betweenness Centrality** - Critical node identification

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Pre-Training (Optional)
The model is already pre-trained, but you can retrain:
```bash
python pretrain.py
```

### Run App
```bash
streamlit run app.py
```

## 📊 Model Performance
- **Architecture**: Gated Graph Convolution Network
- **Training Data**: 200 synthetic urban graphs
- **Accuracy**: 95.5% on validation set
- **Task**: Binary classification (Vulnerable/Resilient)

## 📁 Project Structure
```
Urban-Graph/
├── app.py                  # Streamlit web interface
├── model.py                # UrbanGNN architecture
├── analysis.py             # Graph analysis utilities
├── pretrain.py             # Pre-training script
├── pretrained_model.pth    # Saved checkpoint
└── requirements.txt        # Dependencies
```

## 🎨 Features
- Interactive graph simulation
- Real-time betweenness centrality calculation
- Critical node highlighting
- GNN-based resilience prediction
- Topology visualization with color-coded importance

## 🔬 How It Works
1. **Graph Generation**: Creates random geometric graphs representing urban networks
2. **Centrality Analysis**: Identifies critical nodes using NetworkX algorithms
3. **GNN Inference**: Runs pre-trained model to predict network resilience
4. **Visualization**: Color-coded nodes based on centrality scores

## 🔗 Relevance to ML4SCI
Demonstrates expertise in:
- Geometric Deep Learning
- Graph Neural Networks (GatedGCN)
- Non-Euclidean data structures
- Network analysis and centrality metrics
- Jet-as-Graph classification techniques
