import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalClustering(nn.Module):
    """
    Hierarchical Clustering for task_emb_vec.
    Structure is a tree-based network (TreeLSTM)
    Input: task_emb_vec [batch_size, task_emb_dim], [1,128]
    Output:  root_node HC-embedded vector [batch_size, hidden_dim], [1,128]

    Attributes:
        num_leaf: 4: number of nodes in the first layer, each node [num_task, hidden_dim] [1, 128]
        num_noleaf: 2: number of nodes in the second layer, each node [num_task, hidden_dim]
        input_dim: task_emb_dim, 128
        hidden_dim: tree hidden_dim, 128
        sigma: for assignment softmax cal, 10.0
        update_nets: dict of update net between different nodes.
        assign_net: AssignNet, which contains clustering centers.
    """
    def __init__(self, num_clusters, input_dim=128, hidden_dim=128):
        """
        Args:
            :param num_clusters: 8
            :param input_dim: typically 128.
            :param hidden_dim: typically 128.
        """
        super(HierarchicalClustering, self).__init__()
        self.num_leaf = num_clusters            # 8
        self.num_noleaf = num_clusters // 2     # 4
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.update_nets = nn.ModuleDict()
        for idx in range(self.num_leaf):   # add update nets
            self.update_nets['update_leaf_{}'.format(idx)] = update_block(input_dim, hidden_dim)
        for jdx in range(self.num_noleaf):
            self.update_nets['update_noleaf_{}'.format(jdx)] = update_block(hidden_dim, hidden_dim)
        self.update_nets['update_root'] = update_block(hidden_dim, hidden_dim)
        self.assign_net = AssignNet(self.num_leaf, input_dim, mode='centers')
        if self.num_noleaf > 0:
            self.gate_nets = AssignNet(self.num_noleaf, hidden_dim, mode='dense')   # dense

        self.apply(self.weight_init)    # customized initialization

    def forward(self, inputs):
        """
        Args:
            :param inputs: [batch_size, task_emb_dim], [bs,128]

        Returns:
            :return root_node: HC-embedded vector [batch_size, task_emb_dim], [bs,128]
            :return assigns: similarities in the leaf layer [nodes, bs], [4,bs]
            :return gates: similarities in the noleaf layer  [4,2,bs]
        """
        # layer 0: leaf
        assigns = self.assign_net(inputs)    # [4, bs]
        leaf_nodes = []     # node values in the first (leaf) layer, [4,1,128]
        for idx in range(self.num_leaf):   # update through the first (leaf) layer
            assign = assigns[idx].view(-1, 1)       # [bs, 1]
            updated_inputs = self.update_nets['update_leaf_{}'.format(idx)](inputs)     # [bs, 128]
            leaf_nodes.append(assign * updated_inputs)
        leaf_nodes = torch.stack(leaf_nodes)  # [4,bs,128]

        # layer 1: noleaf
        if self.num_noleaf > 0:
            noleaf_nodes = []     # node values in the second (noleaf) layer, [4,2,bs,128]
            gates = []      # [4, [2, bs]]
            for idx in range(self.num_leaf):    # for each node after leaf layer
                gate = self.gate_nets(leaf_nodes[idx])  # [2, bs]
                gates.append(gate)
                noleaf_nodes_i = []
                for jdx in range(self.num_noleaf):
                    g = gate[jdx].view(-1, 1)       # [bs, 1]
                    updated_inputs = self.update_nets['update_noleaf_{}'.format(jdx)](leaf_nodes[idx])      # [bs, 128]
                    noleaf_nodes_i.append(g * updated_inputs)
                noleaf_nodes_i = torch.stack(noleaf_nodes_i)  # [2, bs, 128]
                noleaf_nodes.append(noleaf_nodes_i)
            gates = torch.stack(gates)  # [4, 2, bs]
            noleaf_nodes = torch.stack(noleaf_nodes)                 # [4,2,bs,128]
            noleaf_nodes = torch.sum(noleaf_nodes, dim=0)     # [2,bs,128]
        else:
            noleaf_nodes = leaf_nodes       # [4, bs, 128]
            gates = torch.ones((self.num_leaf, 1, 1))       # [4, 1, 1]

        # layer 2: root
        root_node = []  # node value in the third (root) layer, [bs,128]
        num_node = self.num_noleaf if self.num_noleaf > 0 else self.num_leaf
        for jdx in range(num_node):
            root_node.append(self.update_nets['update_root'](noleaf_nodes[jdx]))  # [bs,128]
        root_node = torch.stack(root_node)  # [2,bs,128]
        root_node = torch.sum(root_node, dim=0)  # [bs,128]

        return root_node, assigns, gates

    @staticmethod
    def weight_init(m):

        if isinstance(m, nn.Linear):    # for update_blocks and gate_net
            nn.init.xavier_uniform_(m.weight)


def update_block(in_channels, out_channels):
    """
    basic update network block. Tanh(Linear(in, out))
    """
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(in_channels, out_channels)),
        # nn.Linear(in_channels, out_channels),
        nn.Tanh(),
    )


class AssignNet(nn.Module):
    """
    Output assignment softmax probability.
    Input: task_emb_vec [batch_size, task_emb_dim], [1,128]
    Output: assign [num_node, batch_size], [4, 1]

    Attributes:
        cluster_centers: [num_node, task_emb_dim], [4,128]
    """
    def __init__(self, num_node, input_dim, mode='centers'):
        """
        Args:
            :param num_node: same as number of cluster_center, 4; 2 for gates
            :param input_dim: 128.
            :param mode: 'centers' for cluster_centers approach, 'dense' for dense softmax approach.
        """
        super(AssignNet, self).__init__()
        self.num_node = num_node
        self.input_dim = input_dim
        self.mode = mode
        if mode == 'centers':
            self.sigma = 10.0   # temperature for clustering
            self.cluster_centers = nn.Parameter(torch.randn((num_node, input_dim)))
            nn.init.xavier_uniform_(self.cluster_centers)
            # self.register_parameter('cluster_centers', self.cluster_centers)
        elif mode == 'dense':
            self.denses = nn.ModuleList()
            for idx in range(num_node):     # for second layer, 2 dense gates
                self.denses.append(nn.Linear(input_dim, 1))

    def forward(self, inputs):
        """
        Pass batch of task_emb_vec's to get assignment.

        Args:
            :param inputs: [batch_size, task_emb_dim], [bs,128]

        Returns:
            :return assign: assignment probability [num_node, batch_size], [4,bs]
        """
        if self.mode == 'centers':
            assign_batch = []   # 4 * [bs, 128]
            for node in range(self.num_node):   # 4
                assign_batch.append(inputs - self.cluster_centers[node])
            assign_batch = torch.stack(assign_batch)   # [4,bs,128]
            assign = torch.exp(-torch.norm(assign_batch, p=2, dim=2) ** 2 / (2.0*self.sigma))   # [4,bs]
            # assign = torch.exp(-torch.sum(torch.square(assign_batch), dim=2) / (2.0*self.sigma))   # [4,bs]
            assign_sum = torch.sum(assign, dim=0, keepdim=True)     # [1,batch_size], [1,bs]
            assign = assign / assign_sum    # sum(assign, dim=0) == 1
            return assign
        elif self.mode == 'dense':
            assign = []     # num_node*[batch_size,1] 2*[bs]
            for idx in range(self.num_node):
                dense = self.denses[idx]
                assign.append(dense(inputs).squeeze(-1))
            assign = torch.stack(assign)   # [2, bs]
            assign = F.softmax(assign, dim=0)   # i.e., torch.sum(assign, dim=0)==1
            return assign


if __name__ == '__main__':
    net = HierarchicalClustering(8)

    inputs_ = torch.randn(5, 128)
    output_ = net(inputs_)
