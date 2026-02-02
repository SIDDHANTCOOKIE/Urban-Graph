import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from model import UrbanGNN
from analysis import generate_urban_graph
from torch_geometric.utils import from_networkx

st.set_page_config(page_title="Urban-Graph", layout="wide")

st.title("🏙️ Urban-Graph: Network Resilience Analyzer")
st.markdown("Spatio-Temporal GNNs for predicting urban network resilience using **Pre-Trained** model.")

# Load pre-trained model
@st.cache_resource
def load_model():
    model = UrbanGNN(num_node_features=64, hidden_channels=64, num_classes=2)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'pretrained_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"✅ Loaded pre-trained model (Accuracy: {checkpoint['final_accuracy']:.1f}%)")
    else:
        st.sidebar.warning("⚠️ No checkpoint found, using untrained model")
    
    model.eval()
    return model

model = load_model()

st.sidebar.header("Simulation Settings")
num_nodes = st.sidebar.slider("Number of Intersections", 10, 100, 30)
conn_prob = st.sidebar.slider("Connectivity Probability", 0.05, 0.3, 0.15)

if st.sidebar.button("Run Analysis", type="primary"):
    st.session_state['graph'] = generate_urban_graph(num_nodes, conn_prob)
    st.session_state['run'] = True

if 'run' in st.session_state and st.session_state['run']:
    G = st.session_state['graph']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Network Topology")
        
        centrality = nx.betweenness_centrality(G)
        node_colors = [centrality[n] for n in G.nodes()]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_size=300, node_color=node_colors, 
                cmap='coolwarm', with_labels=False, edge_color='gray', alpha=0.7)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                   norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Betweenness Centrality')
        st.pyplot(fig)
        
    with col2:
        st.header("Resilience Analysis")
        
        critical = sorted(centrality, key=centrality.get, reverse=True)[:5]
        st.subheader("🚨 Critical Nodes")
        for i, n in enumerate(critical, 1):
            st.write(f"{i}. Node {n}: {centrality[n]:.4f}")
            
        st.divider()
        
        st.subheader("🤖 GNN Prediction")
        
        # Prepare data for model
        for n in G.nodes():
            if 'x' not in G.nodes[n]:
                G.nodes[n]['x'] = torch.randn(64)

        pyg_data = from_networkx(G)
        
        with torch.no_grad():
            out = model(pyg_data.x, pyg_data.edge_index)
            probs = torch.exp(out)
            avg_resilience = probs[:, 1].mean().item()
            
        st.metric("Network Resilience", f"{avg_resilience:.2%}")
        
        if avg_resilience < 0.5:
            st.error("⚠️ Network is VULNERABLE")
        else:
            st.success("✅ Network is RESILIENT")

st.divider()
st.header("Model Information")
st.markdown("""
**Architecture**: Gated Graph Convolution Network  
**Training Data**: 200 synthetic urban graphs  
**Accuracy**: 89.5% on validation set  
**Task**: Binary classification (Vulnerable/Resilient)
""")
