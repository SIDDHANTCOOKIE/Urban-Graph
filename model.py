import torch
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, GlobalAttention

class UrbanGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, num_layers=3):
        super(UrbanGNN, self).__init__()
        
        # Gated Graph Convolution for temporal/sequential propagation
        self.conv1 = GatedGraphConv(out_channels=hidden_channels, num_layers=num_layers)
        
        # Readout layer (Attention-based global pooling)
        self.attn = GlobalAttention(gate_nn=torch.nn.Linear(hidden_channels, 1))
        
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        """
        x: Node features [num_nodes, num_node_features]
        edge_index: Graph connectivity [2, num_edges]
        """
        # GatedGCN Step
        # Note: GatedGraphConv expects input `x` to match hidden_channels or be projected
        # Here assuming pre-processed or adding a linear projection first if needed.
        # For simplicity in this demo, we assume input dim == hidden dim or use a separate projection.
        
        h = self.conv1(x, edge_index)
        
        # Classification / Regression on node level or graph level
        # This example assumes node-level prediction (e.g. congestion level per intersection)
        out = self.lin(h)
        
        return F.log_softmax(out, dim=1)

if __name__ == "__main__":
    model = UrbanGNN(num_node_features=64, hidden_channels=64, num_classes=5)
    print("UrbanGNN initialized successfully.")
    print(model)
