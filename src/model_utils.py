import os
import warnings
import numpy as np
import sklearn.linear_model as skl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import MessagePassing


class Swish(nn.Module):
    """
    Swish activation function.

    Attributes:
        beta (float): The beta scaling parameter to use. Defaults to 1.
    """
    beta = 1

    def forward(self, x):
        """
        Forward pass of the function.
        """
        return x * torch.sigmoid(self.beta * x)


class CustomConv(MessagePassing):
    """
    Custom implementation of a Graph Convolutional Network (GCN) layer using PyTorch Geometric's MessagePassing class.
    Attributes:
        lin (torch.nn.Linear): The linear layer to compute the node embeddings.
        lin_self (torch.nn.Linear): The linear layer to compute the skip connection.
        nonlin (torch.nn.Module): The non-linear activation function to use. Defaults to ReLU.
        dropout (float): The dropout probability to use. Defaults to 0.2.
    """
    def __init__(self, in_channels, out_channels, nonlin='ReLU', dropout=.2):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.nonlin = eval('nn.' + nonlin)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the GCN layer.
        Args:
            x (Tensor): Node feature matrix of shape [N, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, E].
            edge_attr (Tensor, optional): Edge feature matrix of shape [E, edge_attr_dim]. Default is None.
        Returns:
            Tensor: The output node feature matrix of shape [N, out_channels].
        """
        # Add/Remove self-loops to/from the adjacency matrix.
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        
        # Skip connection with a hidden layer
        self_x = self.nonlin(self.lin_self(x))
        
        # Transform node feature matrix - this will be used as messages in the message passing routine
        x = self.nonlin(F.dropout(self.lin(x), p=self.dropout, training=self.training))
        
        # Compute normalizations as per GCN paper (via normalized graph laplacian)
        row, col = edge_index
        deg = pyg_utils.degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # self.propagate will call self.message for all nodes (effectively creating the computation graphs based on edge_index)
        return self_x + self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index, norm, edge_attr):
        """
        Message computation of the GCN layer.
        Args:
            x_i (Tensor): Source node feature matrix of shape [E, in_channels].
            x_j (Tensor): Target node feature matrix of shape [E, in_channels].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, E].
            norm (Tensor): Normalization coefficients of shape [E, 1].
            edge_attr (Tensor, optional): Edge feature matrix of shape [E, edge_attr_dim]. Default is None.
        Returns:
            Tensor: The computed messages of shape [E, out_channels].
        """
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        """
        Update computation of the GCN layer before returning message passing result.
        Args:
            aggr_out (Tensor): Aggregated output node feature matrix of shape [N, out_channels].
            x (Tensor): Node feature matrix of shape [N, in_channels].
        Returns:
            Tensor: The updated node feature matrix of shape [N, out_channels].
        """
        # Apply L2 norm here
        aggr_out = F.normalize(aggr_out, p=2)
        return aggr_out


class GraphLIME:
    """
    GraphLIME is a class that implements the GraphLIME algorithm for explaining node predictions in graph neural networks.
    The algorithm is based on the LIME (Local Interpretable Model-Agnostic Explanations) framework and uses a kernel-based approach to generate explanations.
    Adapted from GitHub user WilliamCCHuang's implementation.

    Attributes:
        model (torch.nn.Module): The PyTorch model to explain.
        hop (int): The number of hops to use when extracting the subgraph around the node to explain.
        gram_graph (int): Whether to use the graph structure as a Gram matrix. Defaults to 0.
        temp (float): The temperature to use when computing the probabilities. Defaults to 0.
        rho (float): The regularization strength to use when fitting the linear model. Defaults to 0.1.
        positive (bool): Whether to enforce positivity constraints on the linear model coefficients. Defaults to True.
        max_iter (int): The maximum number of iterations to use when fitting the linear model. Defaults to 500.
        fit_method (str): The method to use when fitting the linear model. Defaults to 'LassoLars'.
        tol (float): The tolerance to use when fitting the linear model. Defaults to 1e-5.
        std_factor (float): The factor to use when computing the standard deviation of the kernel. Defaults to 0.1.
        std_type (int): The type of standard deviation to use when computing the kernel. Defaults to 0.
        random_state (int): The random state to use when fitting the linear model. Defaults to None.
        cached (bool): Whether to cache the model predictions. Defaults to True.
        cached_result (torch.Tensor): The cached model predictions.
        contor (int): The number of times the model has been fit.
        sum (float): The sum of the R2 scores of the linear model.
    """
    def __init__(self, model, hop=2, gram_graph=0, temp=0, rho=0.1, positive=True, max_iter=500, fit_method='LassoLars', tol=1e-5, std_factor=0.1, 
                 std_type=0, random_state=None, cached=True):
        self.model = model
        self.hop = hop
        self.gram_graph = gram_graph
        self.temp = temp
        self.rho = rho
        self.positive = positive
        self.max_iter = max_iter
        self.fit_method = getattr(skl, fit_method)
        self.tol = tol
        self.std_factor = std_factor
        self.std_type = std_type
        self.random_state = random_state
        self.cached = cached
        self.cached_result = None
        self.contor = 0
        self.sum = 0
        self.model.eval()

    def _flow(self):
        """
        Returns the message flow direction of the first MessagePassing module in the model.
        If no MessagePassing module is found, returns 'source_to_target' by default.
        """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow

        return 'source_to_target'

    def _subgraph(self, node_idx, x, y, edge_index, **kwargs):
        """
        Extracts a k-hop subgraph around the given `node_idx` from the input graph
        defined by `x` and `edge_index`, and returns the subgraph as well as some
        additional information.

        Args:
            node_idx (int): The index of the central node around which to extract
                    the subgraph.
            x (Tensor): The node feature matrix of shape `(num_nodes, num_node_features)`.
            y (Tensor): The target labels of shape `(num_nodes, num_classes)`.
            edge_index (LongTensor): The edge index matrix of shape `(2, num_edges)`.
            **kwargs: Additional keyword arguments that may contain tensors of shape
                    `(num_nodes, ...)`, `(num_edges, ...)`, or other shapes.

        Returns:
            Tuple[Tensor, Tensor, LongTensor, LongTensor, Tensor, Dict[str, Tensor]]: A tuple
            containing the following elements:
                    - `x`: The node feature matrix of the extracted subgraph, of shape
                        `(num_subgraph_nodes, num_node_features)`.
                    - `y`: The target labels of the extracted subgraph, of shape
                        `(num_subgraph_nodes, num_classes)`.
                    - `edge_index`: The edge index matrix of the extracted subgraph, of shape
                        `(2, num_subgraph_edges)`.
                    - `mapping`: A tensor of shape `(num_subgraph_nodes,)` that maps the node
                        indices of the subgraph back to the original graph.
                    - `edge_mask`: A boolean tensor of shape `(num_edges,)` that indicates which
                        edges in the original graph are present in the subgraph.
                    - `kwargs`: A dictionary of additional keyword arguments, where tensors of
                        shape `(num_nodes, ...)` have been sliced to shape `(num_subgraph_nodes, ...)`
                        and tensors of shape `(num_edges, ...)` have been sliced to shape
                        `(num_subgraph_edges, ...)`.
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = pyg_utils.k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self._flow())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def _init_predict(self, x=None, edge_index=None, logits=None, **kwargs):
        """
        Initializes the predictions of the model.

        Args:
            x (torch.Tensor, optional): The input features tensor. Defaults to None.
            edge_index (torch.Tensor, optional): The graph connectivity tensor. Defaults to None.
            logits (torch.Tensor, optional): The logits tensor. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the model's `predict` method.

        Returns:
            torch.Tensor: The predicted probabilities tensor.

        Raises:
            RuntimeError: If the cached result is not of the same size as the input.
        """
        update_cache = True
        if logits is None:
            if self.cached and self.cached_result is not None:
                if x.size(0) != self.cached_result.size(0):
                    raise RuntimeError(
                        'Cached {} number of nodes, but found {}.'.format(
                            x.size(0), self.cached_result.size(0)))
                update_cache = False
            else:
                with torch.no_grad():
                    logits = self.model.predict(x=x, edge_index=edge_index, **kwargs)
        if update_cache:
            # special case for self.temp in which all nodes are part of the same distribution
            if self.temp:
                temp = abs(self.temp) * (logits.max() - logits.mean()) if self.temp < 0 else self.temp
                probas = (logits / temp).softmax(dim=-1)
            else:
                probas = logits
            # for binary/single output
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)

            self.cached_result = probas

        return self.cached_result
    
    def _compute_kernel(self, x, edge_index=None, edge_attr=None, reduce=False):
        """
        Computes the kernel matrix for the given input features `x`.

        Args:
            x (numpy.ndarray): Input features of shape `(n, d)`.
            edge_index (numpy.ndarray, optional): Edge indices of shape `(2, m)`.
            edge_attr (numpy.ndarray, optional): Edge attributes of shape `(m, k)`.
            reduce (bool, optional): Whether to reduce the kernel matrix to a vector.

        Returns:
            numpy.ndarray: The kernel matrix of shape `(n, n, 1)` or `(n, n, d)`, depending on `reduce`.
        """
        assert x.ndim == 2, x.shape
        
        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        if self.std_type == 0:
            std = np.sqrt(n)
        elif self.std_type == 1:
            std = np.std(x)
        elif self.std_type == 2:
            std = np.std(np.linalg.norm(dist, axis = -1))
        
        K = np.exp(-dist / (2 * std ** 2 * self.std_factor + 1e-10))  # (n, n, 1) or (n, n, d)
        
        if self.gram_graph and edge_index is not None:
            if self.gram_graph == 1:
                edge_attr = None
            adj = pyg_utils.to_dense_adj(edge_index, edge_attr=edge_attr)[0].cpu().numpy() + np.eye(n)
            K = K * np.expand_dims(adj, -1)
            
        return K
    
    def _compute_gram_matrix(self, x):
        """
        Computes the Gram matrix of the input tensor x.

        Args:
            x (numpy.ndarray): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            numpy.ndarray: Gram matrix of shape (batch_size, num_channels, num_channels).
        """
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)
        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G
        
    def explain_node(self, node_idx, x, edge_index, logits=None, **kwargs):
        """
        Computes the GraphLIME explanation for a given node in the graph.

        Args:
            node_idx (int): The index of the node to explain.
            x (torch.Tensor): The node features tensor of shape (num_nodes, num_node_features).
            edge_index (torch.Tensor): The edge index tensor of shape (2, num_edges).
            logits (torch.Tensor, optional): The model's logits tensor of shape (num_nodes, num_classes).
                If None, the model's forward method will be called to obtain the logits.
            **kwargs: Additional keyword arguments to be passed to the model's forward method.

        Returns:
            tuple - Tuple of two numpy arrays:
            - The coefficients of the linear model used to explain the node, of shape (num_node_features,).
            - The mean node features of the subgraph used to explain the node, of shape (num_node_features,).
        """
        probas = self._init_predict(x, edge_index, logits, **kwargs)

        x, probas, edge_index, _, _, kwargs = self._subgraph(
            node_idx, x, probas, edge_index, **kwargs)
        edge_attr = kwargs.get('edge_attr', None)
        if edge_attr is not None and edge_attr.ndim == 2:
            edge_attr = edge_attr[:, 0]

        x = x.detach().cpu().numpy().astype(np.float64)  # (n, d)
        y = probas.detach().cpu().numpy().astype(np.float64)  # (n, classes)

        n, d = x.shape

        K = self._compute_kernel(x, edge_index=edge_index, edge_attr=edge_attr, reduce=False)  # (n, n, d)
        L = self._compute_kernel(y, edge_index=edge_index, edge_attr=edge_attr, reduce=True)  # (n, n, 1)
        K_bar = self._compute_gram_matrix(K)  # (n, n, d)
        L_bar = self._compute_gram_matrix(L)  # (n, n, 1)
        K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
        L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)
        K_fit = K_bar * n
        L_fit = L_bar * n

        warnings.filterwarnings("ignore", category=FutureWarning)
        # some solvers use a Cholesky pivot rather than a tolerance parameter, so make the process solver agnostic
        try:
            solver = self.fit_method(self.rho, positive=self.positive, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state, fit_intercept=False, normalize=False)
        except TypeError:
            solver = self.fit_method(self.rho, positive=self.positive, max_iter=self.max_iter, eps=self.tol, random_state=self.random_state, fit_intercept=False, normalize=False)
        solver.fit(K_fit, L_fit)
        score = solver.score(K_fit, L_fit)
        self.sum += score
        self.contor += 1
        ### Following block can be used to assess the GraphLIME model fit
        # if self.sum / self.contor < 0.18:
        #     raise ValueError('Average R2 too small!')
        # if self.contor:
        #     print(f'Explain count: {self.contor}, Fit {node_idx=} with {n=} perturbations. R2: {score}, Avg R2: {self.sum / self.contor}')
        # print([round(i, 3) for i in solver.coef_], [round(i, 3) for i in x.mean(axis=0)], sep='\n')
        return solver.coef_, x.mean(axis=0)
    

def save_model(model, path='saved/ckp/rl.pt'):
    """
    Saves the state dictionary of a PyTorch model to a specified file path.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The file path to save the model state dictionary to. Defaults to 'saved/ckp/rl.pt'.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model(model, path='saved/ckp/rl.pt', strict=True):
    """
    Loads a saved PyTorch model from a given path.

    Args:
        model (torch.nn.Module): The PyTorch model to load the saved state_dict into.
        path (str): The path to the saved state_dict file.
        strict (bool): Whether to strictly enforce that the keys in the saved state_dict match the keys in the model.

    Returns:
        The PyTorch model with the loaded state_dict.
    """
    if not os.path.exists(path):
        raise ValueError(f'Path {path} does not exist!')
    model.load_state_dict(torch.load(path), strict=strict)
    return model