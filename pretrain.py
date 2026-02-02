"""
Pre-training script for Urban-Graph
Trains the GNN on synthetic graph data and saves the checkpoint
"""
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from model import UrbanGNN
from tqdm import tqdm
import random

def generate_labeled_graph(num_nodes=30, prob=0.15):
    """Generate a graph with resilience labels"""
    G = nx.erdos_renyi_graph(num_nodes, prob)
    
    # Add node features
    for node in G.nodes():
        G.nodes[node]['x'] = torch.randn(64)
    
    # Calculate resilience based on connectivity
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    centrality = nx.betweenness_centrality(G)
    max_centrality = max(centrality.values()) if centrality else 0
    
    # Label: 0 = Vulnerable, 1 = Resilient
    # Vulnerable if poorly connected OR has single points of failure
    is_vulnerable = (avg_degree < 3) or (max_centrality > 0.5)
    G.graph['y'] = 0 if is_vulnerable else 1
    
    return G

def create_dataset(num_graphs=100):
    """Create dataset of labeled graphs"""
    graphs = []
    for _ in range(num_graphs):
        num_nodes = random.randint(20, 50)
        prob = random.uniform(0.05, 0.25)
        G = generate_labeled_graph(num_nodes, prob)
        
        # Convert to PyG Data
        data = from_networkx(G)
        # Manually add graph-level label
        data.y_graph = torch.tensor([G.graph['y']], dtype=torch.long)
        graphs.append(data)
    
    return graphs

def train(epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    # Create dataset
    print("Generating training dataset...")
    dataset = create_dataset(num_graphs=200)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    model = UrbanGNN(num_node_features=64, hidden_channels=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    print("Starting pre-training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index)
            
            # Aggregate node predictions to graph level (mean pooling)
            probs = torch.exp(out)  # log_softmax -> probs
            graph_pred = probs[:, 1]  # Probability of "Resilient"
            
            # Simple aggregation: average resilience of nodes
            batch_size = batch.y_graph.shape[0]
            nodes_per_graph = batch.x.shape[0] // batch_size
            
            graph_preds = []
            for i in range(batch_size):
                start_idx = i * nodes_per_graph
                end_idx = (i + 1) * nodes_per_graph
                avg_pred = graph_pred[start_idx:end_idx].mean()
                graph_preds.append(avg_pred)
            
            graph_preds = torch.stack(graph_preds)
            
            # Binary classification loss
            labels = batch.y_graph.float()
            loss = F.binary_cross_entropy(graph_preds, labels)
            
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            predicted = (graph_preds > 0.5).long()
            correct += (predicted == batch.y_graph).sum().item()
            total += batch_size
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%")
    
    # Save checkpoint
    print("Saving checkpoint...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'final_accuracy': accuracy
    }, 'pretrained_model.pth')
    
    print("Pre-training complete!")

if __name__ == "__main__":
    train()
