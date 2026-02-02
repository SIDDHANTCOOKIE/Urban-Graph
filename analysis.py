import torch
import networkx as nx
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from model import UrbanGNN

def generate_urban_graph(num_nodes=50, prob=0.1):
    """
    Simulates an urban road network using a random geometric graph or Erdős-Rényi.
    """
    G = nx.erdos_renyi_graph(num_nodes, prob)
    # Add dummy node features
    for node in G.nodes():
        G.nodes[node]['x'] = torch.randn(64) # 64-dim feature vector
    return G

def analyze_resilience(G):
    print("Calculating Betweenness Centrality...")
    centrality = nx.betweenness_centrality(G)
    
    # Identify critical nodes (top 5)
    critical_nodes = sorted(centrality, key=centrality.get, reverse=True)[:5]
    print(f"Top 5 Critical Nodes (Potential Bottlenecks): {critical_nodes}")
    
    return critical_nodes

def run_prediction():
    # 1. Generate Data
    G = generate_urban_graph()
    
    # 2. Analyze
    analyze_resilience(G)
    
    # 3. Convert to PyG Data
    pyg_data = from_networkx(G)
    
    # 4. Run Model
    model = UrbanGNN(num_node_features=64, hidden_channels=64, num_classes=2) # 2 classes: Resilient, Vulnerable
    
    model.eval()
    with torch.no_grad():
        # Input features come from the 'x' attribute we set earlier
        out = model(pyg_data.x, pyg_data.edge_index)
        pred = out.argmax(dim=1)
        
    print(f"Model Predictions for {pyg_data.num_nodes} nodes: {pred[:10]}... (showing first 10)")

if __name__ == "__main__":
    run_prediction()
