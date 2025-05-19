import torch.nn as nn
import torch.nn.functional as F
import torch

# simple graph convolutional layer (without normalization of ad or separately implemented self-loops)
# showed even better performance than more complex GCNs and trains faster
# we do not model very big graphs so that simple gcn still works well, fast and shows the advantage of adversarial training
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # initialize the weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        # initialize the bias to zero
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        ax = torch.matmul(adj, x)
        out = torch.matmul(ax, self.weight) + self.bias
        return out

# for DQN it was chosen to use better the MLP structure
class DQN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, node_features):
        q_values = self.fc(node_features)
        return q_values

class A2C(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GCNLayer(in_features=1, out_features=hidden_dim)
        self.gcn2 = GCNLayer(in_features=hidden_dim, out_features=hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, num_nodes)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, adj, current_node):
        # change the shape of node_features to [num_nodes, 1]
        x = node_features.squeeze(0).unsqueeze(-1)

        h = F.relu(self.gcn1(x, adj))
        h = self.gcn2(h, adj)

        # print(f"Shape of h is {h.shape}")
        # extracts tensor of features for the current node
        node_emb = h[current_node]
        # print(f"Tensor of features for current node {node_emb}")
        # policy value (logits) from actor
        logits = self.policy_head(node_emb)
        # value of critic
        # converts to scalar tensor
        value = self.value_head(node_emb).squeeze(-1)
        return logits, value


