import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import logging
from torch_geometric.nn import GCNConv, GATConv

logger = logging.getLogger(__name__)

class GraphSAGE(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GraphSAGE, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_out)
        self.dim_h = dim_h

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        device = x.device

        # Add self-loops
        self_loops = torch.stack([torch.arange(num_nodes, device=device)] * 2, dim=0)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
            self_weight = torch.ones(num_nodes, device=device)
        else:
            self_weight = torch.ones(num_nodes, device=device) * edge_weight.mean()
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_weight = torch.cat([edge_weight, self_weight], dim=0)

        # Adjacency matrix (sparse)
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            (num_nodes, num_nodes)
        )

        # Normalize adjacency (row-wise)
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        degree = degree.clamp(min=1.0)
        norm = 1.0 / degree
        norm_adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight * norm[edge_index[0]],
            (num_nodes, num_nodes)
        )

        # Layer 1: Mean aggregation
        h = torch.sparse.mm(norm_adj, x)
        h = self.linear1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        # Layer 2
        h = torch.sparse.mm(norm_adj, h)
        h = self.linear2(h)
        return F.log_softmax(h, dim=-1)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)
        self.dim_h = dim_h

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return F.log_softmax(h, dim=-1)

class GAT(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_in, dim_h, heads=heads, dropout=0.2)
        self.conv2 = GATConv(dim_h * heads, dim_out, heads=1, concat=False, dropout=0.2)
        self.dim_h = dim_h
        self.heads = heads

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=-1)

def train(model, data, epochs=200, lr=0.005, class_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        if class_weights is not None:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights.to(data.x.device))
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Training accuracy
        pred = out.argmax(dim=1)
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.size(0)
        if epoch % 50 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}")
    return model

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        pred = out.argmax(dim=1)
        test_mask = data.test_mask[data.test_mask < data.x.size(0)]
        if test_mask.size(0) == 0:
            logger.warning("Empty test mask, returning zero accuracy")
            return 0.0, [], 0.0
        acc = pred[test_mask].eq(data.y[test_mask]).sum().item() / test_mask.size(0)
        cm = confusion_matrix(data.y[test_mask].cpu(), pred[test_mask].cpu())
        kappa = cohen_kappa_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
        logger.info(f"Test Accuracy: {acc:.4f}, Cohen Kappa: {kappa:.4f}")
    return acc, cm, kappa