import inspect
import itertools
import random

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_sparse import SparseTensor

from .model_utils import Swish, CustomConv


GNN_LAYERS = {
    "GCN1": pyg_nn.GCNConv, # uses edge_weight
    "GraphConv": pyg_nn.GraphConv, # uses edge_weight
    "SAGE": pyg_nn.SAGEConv, # uses just a mean aggregator
    "GAT1": pyg_nn.GATConv, # latest version uses edge_attr for attention
    "GAT2": pyg_nn.GATv2Conv, # improved version of GAT1
    "GIN1": pyg_nn.GINConv, # as powerful as 1-WL, NO edge attr
    "GINE": pyg_nn.GINEConv, # uses edge_attr, same len as x
    "NNConv": pyg_nn.NNConv, # uses edge_attr of variable length
    "FAConv": pyg_nn.FAConv, # uses edge_weight in attention
    "RGCN": pyg_nn.RGCNConv, # uses edge_type
    "GEN": pyg_nn.GENConv, # uses MessageNorm internally AND edge_attr
    "Custom": CustomConv, # can customize to use edge_attr
}


class Model(nn.Module):
    """
    A PyTorch base template for creating node ranking graph neural network models.

    Attributes:
        device (torch.device): The device on which the model is loaded.

    Methods:
        from_dict: A static method that creates a new instance of the model from a dictionary of arguments.
        load_state_dict: Loads the state dictionary of the model.
        to: Moves the model to the specified device, and updates the `device` attribute.
        cpu: Moves the model to the CPU, and updates the `device` attribute.
        cuda: Moves the model to the GPU, and updates the `device` attribute.
    """    
    @staticmethod
    def from_dict(k_hops=2, static_measures=('degree',), n_models=1, initial_weights='custom', **kwargs):
        """
        Creates a new instance of a ranking model from a dictionary of hyperparameters.

        Args:
            k_hops (int): The number of hops for which infectious neighborhood features were computed
            static_measures (tuple): A tuple of strings representing the static network measures to use.
            n_models (int): The number of models to ensemble.
            initial_weights (str): The path to the initial weights file, or 'custom' to use the initialization scheme of `init_weights`.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            A new instance of a ranking model.
        """
        # input features will be 4 (prev_pos, prev_neg, prev_untested, ever_pos) + k_hops + num of static measures
        input_features = 4 + k_hops
        for measure in static_measures:
            try:
                input_features += int(measure.split(':')[1])
            except IndexError:
                input_features += 1
        model_cls = EnsembleModel if n_models > 1 else RankingModel
        model = model_cls(input_features, 1, n_models=n_models, **kwargs)
        if initial_weights == 'custom':
            model.apply(init_weights)
        elif initial_weights:
            ckp_f = torch.load(initial_weights)
            try:
                model.load_state_dict(ckp_f['model_state_dict'])
            except KeyError:
                model.load_state_dict(ckp_f)
        # for logging purposes
        model.init_kwargs = dict(k_hops=k_hops, static_measures=static_measures, n_models=n_models, initial_weights=initial_weights, **kwargs)
        return model 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cpu')
        
    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the state dictionary of the model. This is adapting the original implementation to accommodate checkpoints from older versions of torch_geometric,
        which had a bug in the LayerNorm module that caused the in_channels to be 1 instead of the actual number of input channels.

        Args:
            state_dict (dict): The state dictionary to load.
            strict (bool, optional): Whether to strictly enforce that the keys in the state dictionary match the keys in the model. Defaults to True.
        """
        # make checkpoints compatible to newer versions of torch_geometric that fixed the LayerNorm in_channels issue
        for m_name, m in self.named_modules():
            if isinstance(m, pyg_nn.LayerNorm):
                for param in ('weight', 'bias'):
                    param_name = f'{m_name}.{param}'
                    param_val = state_dict[param_name]
                    if param_val.numel() == 1:
                        state_dict[param_name] = torch.ones_like(m.state_dict()[param], device=self.device) * param_val.to(self.device)
        super().load_state_dict(state_dict, strict=strict)

    def cpu(self):
        """
        Move the model to CPU and update the `device` attribute.

        Returns:
            The model on CPU.
        """
        on_cpu = super().cpu()
        # Update the device attribute
        self.device = torch.device('cpu')
        return on_cpu

    def cuda(self, device=None):
        """
        Moves the model to the specified CUDA device, updating the `device` attribute.

        Args:
            device (int, str, torch.device, optional): The device to move the model to.
                If None, the default CUDA device is used. If an integer is provided,
                the corresponding CUDA device is used. If a string is provided, it
                should be in the format 'cuda:<device_id>'. If a torch.device object
                is provided, it is used directly. Defaults to None.

        Returns:
            The model itself, after being moved to the specified device.
        """
        on_cuda = super().cuda(device)
        # Update the device attribute
        if device is None:
            self.device = torch.device('cuda')
        elif isinstance(device, int):
            self.device = torch.device(f'cuda:{device}')
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f'Invalid device type: {type(device)}')
        return on_cuda
    
    def to(self, device=None, **kwargs):
        """
        Moves and/or casts the parameters and buffers to device.

        Args:
            device (int, str, torch.device, optional): the desired device of the parameters and buffers.
                If None, it will be the same device as the current module.
                If int, it will correspond to the index of the CUDA device.
                If str, it will correspond to the name of the CUDA device.
            **kwargs: any additional keyword arguments to be passed to the underlying to() method.

        Returns:
            Module: self, with the parameters and buffers moved and/or cast to device.
        """
        on_device = super().to(device, **kwargs)
        # Update the device attribute
        if isinstance(device, int):
            self.device = torch.device(f'cuda:{device}')
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        elif device is not None:
            raise TypeError(f'Invalid device type: {type(device)}')
        return on_device
            

class EnsembleModel(Model):
    """
    A model that combines the predictions of multiple RankingModel instances.

    Attributes:
        models (nn.ModuleList): The list of RankingModel instances.
        n_models (int): The number of models.
        combine (str): The name of the method used to combine the predictions.
        avg (bool): Whether the predictions are normalized by the number of models upon combination.
            This is True if `combine` contains the word 'avg', and False otherwise.
        weight (nn.Parameter or list): The weights used to combine the predictions. 
            If `combine` contains the word 'weight', this is a nn.Parameter instance. Otherwise, it is a list of ones.

    Methods:
        __getitem__(self, i): Returns the i-th RankingModel instance.
        predict(self, *args, i_model=-1, **kwargs): Computes the node-level prediction of the i-th model (i.e. `y_pred`). 
            If i_model=-1, computes the prediction of the ensemble.
        forward(self, *args, **kwargs): Computes the prediction (`y_pred`, `v_score`) of the ensemble.
        update_device(self): Updates the device used by the models.
    """    
    def __init__(self, input_dim, output_dim=1, n_models=3, combine='avg', torch_seed=None, **kwargs):
        super().__init__()
        base_seed = random.randint(0, 1e9) if torch_seed is None else torch_seed
        self.models = nn.ModuleList([RankingModel(input_dim, output_dim, torch_seed=base_seed + 3 * i_model, **kwargs) for i_model in range(n_models)])
        self.n_models = n_models
        self.combine = combine
        self.avg = combine.__contains__('avg')
        self.weight = nn.Parameter(torch.ones(2 * n_models).reshape(-1, 1)) if combine.__contains__('weight') else [1] * (2 * n_models)
        
    def __getitem__(self, i):
        """
        Get the i-th model in the list of models.

        Args:
            i (int): The index of the model to retrieve.

        Returns:
            model: The i-th model in the list of models.
        """
        return self.models[i]
    
    @property
    def h_prev(self):
        """
        Returns the previous node hidden state of the first model in the list of `models`.
        Since the setter sets the previous hidden state of all models to the same tensor, we can return the hidden state of either model here.

        Returns:
            torch.Tensor: The hidden state of the first model in the list of models.
        """
        return self.models[0].h_prev
    
    @h_prev.setter
    def h_prev(self, value):
        """
        Sets the hidden state of each model in the ensemble to the given value.

        Args:
            value: A tensor representing the new hidden state.

        Returns:
            None
        """
        for model in self.models:
            model.h_prev = value
            
    @property
    def deterministic(self):
        """
        Returns whether the model is deterministic or not. All models in the ensemble are assumed to have the same value for this attribute.

        Returns:
            bool: Whether the model is deterministic or not.
        """
        return self.models[0].deterministic
    
    @property
    def detach_h(self):
        """
        Returns whether the hidden state of the model is to be detached after each pass. All models in the ensemble are assumed to have the same value for this attribute.

        Returns:
            bool: Whether the hidden state of the model is to be detached after each pass.
        """
        return self.models[0].detach_h
        
    def predict(self, *args, i_model=-1, **kwargs):
        """
        Computes the node-level prediction of the i-th model. If i_model=-1, computes the prediction of the ensemble.

        Args:
            *args: Variable length argument list.
            i_model (int): Index of the model to use for prediction. If not provided, the default `-1` is used, which represents the ensemble.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The predicted `y_pred` output of the model `i_model` for the given input.
        """
        model = self.models[i_model] if i_model >= 0 else self
        return model.forward(*args, **kwargs)[0]
            
    def forward(self, *args, **kwargs):
        """
        Forward pass of the ensemble model.
        The models within the ensemble may output only `y_pred`, only `v_score` or both depending on the `scorer_type` argument.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: Tuple of `y_pred` and `v_score`.
        """
        y_pred = v_score = None
        for i_model, model in enumerate(self.models):
            y_pred_i, v_score_i = model(*args, **kwargs)
            if y_pred_i is not None:
                y_pred = y_pred_i if y_pred is None else y_pred + self.weight[i_model] * y_pred_i
            if v_score_i is not None:
                v_score = v_score_i if v_score is None else v_score + self.weight[i_model + 1] * v_score_i

        if self.avg:
            y_pred = y_pred / self.n_models if y_pred is not None else None
            v_score = v_score / self.n_models if v_score is not None else None
            
        return y_pred, v_score


class RankingModel(Model):
    """
    A PyTorch model for ranking nodes in a graph based on their features and connectivity.

    Args:
        input_dim (int): The dimensionality of the input features for each node.
        output_dim (int, optional): The dimensionality of the output scores for each node. Defaults to 1.
        hidden_dim (int, optional): The dimensionality of the hidden layers in the model. Defaults to 64.
        dropout_head (float, optional): The dropout rate to apply to the heads of the MLP/RNN layers. Defaults to 0.1.
        agg (str, optional): The aggregation method to use for the GNN layers. Can be 'cat' or 'add'. Defaults to 'cat'.
        state_score_method (str, optional): The method to use for pooling node states to obtain a single score. Can be 'max', 'mean', or 'sum'. Defaults to 'max'.
        incl_gnn_ins (bool, optional): Whether to include the input features in the GNN layers. Defaults to False.
        last_incl_gnn_outs (bool, optional): Whether to include the output of the GNN layers in the final MLP layer. Defaults to False.
        reduce_h (str, optional): The type of RNN layer to use for reducing the hidden state of the GNN layers. Can be 'gru', 'lstm', or ''. Defaults to ''.
        torch_seed (int, optional): The seed to use for PyTorch random number generation. Defaults to None.
        reset_h_seed (int, optional): The method to use for initializing the hidden state of the GNN layers. Can be -1, 0, 1, or 2. Defaults to 0.
            -1: zeros everywhere but the initial known infected where another integer is used
            <0: zeros everywhere
            0: randn with disabled seed reset
            1: randn with resetting the seed for training
            2: randn with resetting the seed always (full consistency)
        deterministic (bool, optional): Whether to use deterministic algorithms for PyTorch and PyTorchGeometric. Defaults to False.
        detach_h (bool, optional): Whether to detach the hidden state of the GNN layers from the computation graph. Defaults to False.
        force_intern_h (bool, optional): Whether to force the use of an internal hidden state for the GNN layers. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the `GNNModule` constructor.

    Attributes:
        hidden_dim (int): The dimensionality of the hidden layers in the model.
        incl_gnn_ins (bool): Whether to include the input features in the GNN layers.
        last_incl_gnn_outs (bool): Whether to include the output of the GNN layers in the final MLP layer.
        detach_h (bool): Whether to detach the hidden state of the GNN layers from the computation graph.
        reduce_h (str): The type of RNN layer to use for reducing the hidden state of the GNN layers.
        force_intern_h (bool): Whether to force the use of an internal hidden state for the GNN layers.
        info (GNNModule): The information diffusion model.
        diffusion (GNNModule): The diffusion model.
        nonlinear_scale (nn.Parameter): The parameter used to scale the hidden state of the GNN layers.
        g_hidden (nn.Module): The module used to reduce the hidden state of the GNN layers.
        f_scorer (nn.Module): The module used to compute the final scores.
        f_value (nn.Module): The module used to compute the final value.
        h_prev (torch.Tensor): The hidden state of the GNN layers of the previous timestamp.
        mode (int): The ID of the layer type utilized in g_hidden. This is set to 0 for MLP, 1 for GRU, or 2 for LSTM.

    Methods:
        predict(self, *args, **kwargs): Computes the node-level prediction of the model (i.e. `y_pred`).
        forward(self, x, edge_index, edge_attr=None, edge_current=None, batch_idx=None, scorer_type=2, h_prev=None, **kwargs): Computes the full prediction of the model.
    """    
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, dropout_head=.1, agg='cat', state_score_method='max', incl_gnn_ins=False, last_incl_gnn_outs=False, 
                 reduce_h='', torch_seed=None, reset_h_seed=0, deterministic=False, detach_h=False, force_intern_h=False, **kwargs):
        super().__init__()
        self.init_kwargs = {}
        self.deterministic = False
        self.h_seed = None
        self.reset_h_seed = reset_h_seed
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            torch.backends.cudnn.benchmark = False
            # this variable will force deterministic behavior w.r.t. both PyTorch and PyTorchGeometric
            if deterministic:
                torch.use_deterministic_algorithms(True)
                self.deterministic = True
            # use 'minimum' determinism setting otherwise (this does not cause errors if an op is non-deterministic)
            else:
                torch.use_deterministic_algorithms(False)
                torch.backends.cudnn.deterministic = True
            # set a h_seed only if reset_h_seed is enabled
            if reset_h_seed > 0:
                self.h_seed = torch_seed  
    
        self.hidden_dim = hidden_dim
        self.incl_gnn_ins = incl_gnn_ins
        self.last_incl_gnn_outs = last_incl_gnn_outs
        # detach from graph previous h
        self.detach_h = detach_h
        self.reduce_h = reduce_h.lower()
        self.force_intern_h = force_intern_h
        # always mark task as timenode
        kwargs['task'] = 'timenode'
        # instatiate Information module
        # the heads of an GNNModule is removed by setting dropout_head = 1 (i.e. keeps only backbone)
        self.info = GNNModule(input_dim, hidden_dim=hidden_dim, dropout_head=1, deterministic=deterministic, **kwargs)

        # REPLACE kwargs for instantiating Diffusion model
        # the Diffusion model normally has a single GNN layer, and therefore num_layers = 1 and last_fully_adjacent = False
        kwargs['num_layers'] = 1
        kwargs['last_fully_adjacent'] = False
        # the Diffusion model normally utilizes the first dimension of the edge attributes as edge weights (i.e. the actual edge weight of the network)
        kwargs['edge_dim'] = 0
        kwargs['edge_attr_scaling'] = False
        # at this point, one can overwrite any setting in `kwargs` specific for the Diffusion model by supplying `diff_<setting>` in `kwargs`
        for k, arg in kwargs.copy().items():
            if k.startswith('diff_'):
                key_to_replace = k[k.index('_')+1:]
                kwargs[key_to_replace] = arg
        self.diffusion = GNNModule(input_dim, hidden_dim=hidden_dim, dropout_head=1, deterministic=deterministic, **kwargs)
        self.nonlinear_scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
                
        # hidden state of previous timestamp
        self.h_prev = None
        if reduce_h in ('gru', 'lstm'):
            g_hidden_model = RNN
            self.mode = 1 if reduce_h == 'gru' else 2
        else:
            g_hidden_model = MLP
            self.mode = 0
        
        # CREATE MLP/RNN layers
        g_input_dim = f_input_dim = input_dim
        if agg == 'cat':
            # for RNN, h_prev is not concatenated to the features rather being used as the RNN hidden state
            g_input_dim += (3 if self.mode == 0 else 2) * hidden_dim
            # either 4 quantities (h, h_prev, d, i) or 2 quantities (h, h_prev) go into the final MLP
            f_input_dim += (4 if last_incl_gnn_outs else 2) * hidden_dim
        else:
            g_input_dim += hidden_dim
            f_input_dim += hidden_dim
        
        # the dropout of the MLP/RNN layers will be the supplied dropout_head, while the layers will be the layers_head
        self.g_hidden = g_hidden_model(g_input_dim, hidden_dim, dropout_head=dropout_head, agg=agg, rnn_layer=reduce_h, **kwargs)
        # last MLP will always have a linear head (to output_dim), whereas g_hidden supports either having or not having one
        kwargs['linear_head'] = True
        self.f_scorer = MLP(f_input_dim, output_dim, dropout_head=dropout_head, agg=agg, **kwargs)
        # the state-value MLP will have a state_scorer to output a value
        state_scorer = getattr(pyg_nn, f'global_{state_score_method}_pool')
        self.f_value = MLP(f_input_dim, output_dim, dropout_head=dropout_head, agg=agg, state_scorer=state_scorer, **kwargs)
    
    def predict(self, *args, **kwargs):
        """
        Predicts the node-level output of the model for the given input.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The predicted `y_pred` output of the model.
        """
        return self.forward(*args, **kwargs)[0]
        
    def forward(self, x, edge_index, edge_attr=None, edge_current=None, batch_idx=None, scorer_type=2, h_prev=None, **kwargs):
        """
        Computes the forward pass of the rank model.

        Args:
            x (torch.Tensor): The input node features of shape (num_nodes, input_size).
            edge_index (torch.Tensor): The graph connectivity in COO format of shape (2, num_edges).
            edge_attr (torch.Tensor, optional): The edge features of shape (num_edges, edge_size). Defaults to None.
            edge_current (tuple(torch.Tensor, torch.Tensor), optional): The current timestamp graph connectivity and edge features in COO format. Defaults to None.
            batch_idx (torch.Tensor, optional): The batch index of each node of shape (num_nodes,). Defaults to None.
            scorer_type (int, optional): The type of scorer to use. Defaults to 2.
                0: calculate a score for every node
                1: calculate a score for the state
                2: calculate both
            h_prev (torch.Tensor, optional): The previous hidden state of shape (num_nodes, hidden_size). Defaults to None.
            **kwargs: Additional keyword arguments to pass to the diffusion and info modules.

        Returns:
            list(torch.Tensor): A list containing the node scores and the state score.
        """
        if batch_idx is None:
            batch_idx = torch.zeros(len(x), dtype=int, device=self.device)
        if h_prev is None or self.force_intern_h:
            intern_h = True
            # lazily initialize h_prev if it wasn't yet initialized
            if self.h_prev is None:
                shape = [x.shape[0], self.hidden_dim]
                if self.reset_h_seed == -1:
                    # repeat() should get the arguments (1,1?,input_size) to be consistent with both MLP and RNN logic
                    shape[:-1] = [1] * len(shape[:-1])
                    # we repeat the last feature, which specifies whether the node has ever tested positive or not
                    self.h_prev = x[:, -1:].repeat(*shape)
                # when reset_h_seed is negative, assume h_prev initializes to zeros
                elif self.reset_h_seed < 0:
                    self.h_prev = torch.zeros(shape, device=self.device)
                # otherwise, torch.randn will be used, with different seeding logic based on h_seed and reset_h_seed
                else:
                    # if h_seed exists (supplied a torch_seed and a reset_h_seed > 0),
                    # we either always reset the seed (i.e. reset_h_seed = 2, full consistency) or reset just for evaluation (reset_h_seed = 1)
                    if self.h_seed is not None and (self.reset_h_seed == 2 or not self.training):
                        torch.manual_seed(self.h_seed)
                    self.h_prev = torch.randn(shape, device=self.device)
                if self.mode == 2:
                    self.h_prev = (self.h_prev, torch.zeros_like(self.h_prev))
            h_prev = self.h_prev
        else:
            intern_h = False
            
        if edge_current:
            edge_index_current, edge_attr_current = edge_current
        else:
            edge_index_current, edge_attr_current = edge_index, edge_attr
        
        x_gnn = torch.cat((x, h_prev), dim=-1) if self.incl_gnn_ins else x
        d, _ = self.diffusion(x_gnn, edge_index_current, edge_attr_current, batch_idx, **kwargs)
        i, _ = self.info(x_gnn, edge_index, edge_attr, batch_idx, **kwargs)
        
        h = self.g_hidden(x, d, i, h_prev)
        if self.reduce_h.__contains__('l2'):
            h = nn.functional.normalize(h, p=2., dim=-1)
        elif self.reduce_h.__contains__('tan'):
            h = torch.tanh(h)
        elif self.reduce_h.__contains__('sig'):
            h = torch.sigmoid(h)
        elif self.reduce_h.__contains__('mean'):
            h = h - h.mean(dim=0, keepdim=True)
        if self.reduce_h.__contains__('scl'):
            h = self.nonlinear_scale * h
        if self.reduce_h.__contains__('std'):
            h = h / h.std(dim=0, unbiased=False, keepdim=True)
        self.h_prev = h.detach() if self.detach_h else h
            
        inputs = [x, h, h_prev]
        if self.last_incl_gnn_outs:
            if self.last_incl_gnn_outs == 2:
                d = nn.functional.normalize(d, p=2)
                i = nn.functional.normalize(i, p=2)
            inputs += [d, i]
        # holder for node_scores and state_score
        output = [None, None]
        # for score_flag 0 or 2, we calculate a score for every node
        if scorer_type % 2 == 0:
            output[0] = self.f_scorer(*inputs).squeeze(dim=-1)
        # for score_flag 1 or 2, we calculate a score for the state
        if scorer_type >= 1:
            output[1] = self.f_value(*inputs, batch_idx=batch_idx).squeeze(dim=-1)
        
        # this gets executed only if there was no externally supplied h_prev
        if intern_h:
            self.h_prev = h.detach() if self.detach_h else h
        return output
    
    
class RNN(nn.Module):
    """
    Recurrent neural network module that takes multiple inputs, aggregates them and then computes the afferent recurrent output and hidden state.
    
    Args:
        input_dim (int): The number of expected features in the input.
        output_dim (int, optional): The number of output features. Default is 1.
        hidden_dim (int, optional): The number of features in the hidden state. Default is 64.
        dropout_head (float, optional): The dropout probability for the input layer. Default is 0.1.
        agg (str, optional): The aggregation method to use when combining multiple inputs. Can be 'cat' for concatenation or 'sum' for summation. Default is 'cat'.
        rnn_layer (str, optional): The type of RNN layer to use. Can be 'GRU' or 'LSTM'. Default is 'GRU'.
        layers_head (int, optional): The number of RNN layers to use. Default is 1.
        linear_head (bool, optional): Whether to use a linear layer as the output or not. Default is False.

    Attributes:
        agg (str): The aggregation method to use when combining multiple inputs.
        model (nn.Module): The RNN model.
        output (nn.Module): The output layer.
    """    
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, dropout_head=.1, agg='cat', rnn_layer='GRU', layers_head=1, linear_head=False, **kwargs):
        super().__init__()
        self.agg = agg
        rnn_layer = eval(f'nn.{rnn_layer.upper()}Cell')
        self.model = nn.ModuleList([rnn_layer(input_dim, hidden_dim) for _ in range(layers_head)])
        self.output = nn.Linear(hidden_dim, output_dim) if linear_head else nn.Identity()
        
    def forward(self, *args, batch_idx=None):
        """
        Forward pass of the RNN model.

        Args:
            *args: Variable length argument list of tensors to be concatenated or summed before being fed through the layers.
            batch_idx (int, optional): Index of the current batch. Defaults to None.

        Returns:
            tuple: Tuple containing:
                - output tensor of the forward pass.
                - hidden state tensor of the forward pass.
        """
        args, h_prev = args[:-1], args[-1]
        if self.agg == 'cat':
            inp = torch.cat(args, dim=1)
        else:
            inp = torch.zeros_like(args[1])
            for arg in args[1:]:
                inp += arg
            inp = torch.cat((args[0], inp), dim=1)
        for l in self.model:
            h = l(inp, h_prev)
        # for LSTMS, h will be a tuple containing h state and c state (long-term and short-term memories)
        if isinstance(h, tuple):
            return self.output(h[0]), h[1]
        return self.output(h)
    
    
class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) neural network module that takes multiple inputs, aggregates them and then computes the afferent MLP output.
    If state_scorer is provided, the output is passed through it before being returned. This is useful for computing a state score from the node hidden states.

    Args:
        input_dim (int): The number of input features.
        output_dim (int, optional): The number of output features. Defaults to 1.
        hidden_dim (int, optional): The number of hidden units in each layer. Defaults to 64.
        dropout_head (float, optional): The dropout probability for the input layer. Defaults to 0.1.
        agg (str, optional): The aggregation method for the input features. Can be 'cat' or 'sum'. Defaults to 'cat'.
        nonlin_head (str, optional): The activation function for the hidden layers. Defaults to 'ReLU'.
        layers_head (int, optional): The number of hidden layers. Defaults to 1.
        linear_head (bool, optional): Whether to use a linear output layer. Defaults to True.
        state_scorer (callable, optional): A function that outputs a state score from the node hidden states. Defaults to None.

    Attributes:
        agg (str): The aggregation method for the input features.
        model (nn.Module): The MLP model.
        output (nn.Module): The output layer.
        state_scorer (callable): The function that outputs a state score from the node hidden states.
    """
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, dropout_head=.1, agg='cat', nonlin_head='ReLU', layers_head=1, linear_head=True, state_scorer=None, **kwargs):
        super().__init__()
        self.agg = agg
        nonlin = nonlin_head if nonlin_head else kwargs.get('nonlin', 'Identity')
        if nonlin.__contains__('Swish'):
            try:
                Swish.beta = float(nonlin.split(':')[1])
            except (IndexError, ValueError):
                pass
            nonlin = Swish
        else:
            nonlin = eval(f'nn.{nonlin}')
        self.model = nn.Sequential(*itertools.chain.from_iterable((
            (nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
            nn.Dropout(dropout_head),
            nonlin()) for i in range(layers_head))))
                
        self.output = nn.Linear(hidden_dim if layers_head >= 1 else input_dim, output_dim) if linear_head else nn.Identity()
        self.state_scorer = state_scorer
        
    def forward(self, *args, batch_idx=None):
        if self.agg == 'cat':
            inp = torch.cat(args, dim=1)
        else:
            inp = torch.zeros_like(args[1])
            for i, arg in enumerate(args[1:]):
                inp += arg
            inp = torch.cat((args[0], inp), dim=1)
        if self.state_scorer is not None:
            inp = self.state_scorer(inp, batch_idx)
        return self.output(self.model(inp))


class GNNModule(nn.Module):
    """
    General purpose graph neural network module that can be used for node-level, edge-level of graph-level prediction tasks.

    Args:
        input_dim (int): The dimensionality of the input node features.
        output_dim (int, optional): The dimensionality of the output node features. Defaults to 1.
        hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 64.
        transform_pre_mp (bool, optional): Whether to apply a linear transformation to the input features before message passing. Defaults to False.
        num_layers (int, optional): The number of message passing layers. Defaults to 2.
        nonlin (str or callable, optional): The nonlinearity to use across the model. Defaults to 'ReLU'.
        norm_layer (callable, optional): The normalization layer to use across the model. Defaults to None.
        dropout (float, optional): The dropout probability to use across the model. Defaults to 0.2.
        dropout_head (float or None, optional): The dropout probability to use after the final message passing layer. 
            If None, defaults to the value of `dropout`. Defaults to None.
        task (str, optional): The type of prediction task. Defaults to 'node'. Can be one of:
            - 'node': node-level prediction task; this signals to caller that a semi-supervised node prediction task is to be performed (so node masking is to be applied)
            - 'timenode': node-level prediction task over time; equivalent to 'node' for the purpose of this module, but signals to caller that the model is to be used for
                time-series node predictions rather than inductive semi-supervised node predictions (i.e. no masking needed)
            - 'edge': edge-level prediction task; this allows caller to calculate a loss based on the node latent space instead of computing a supervised loss using the outputs
                This is the natural choice for edge-level models based on GAE/VGAE.
            - 'graph': graph-level prediction task; this leads to a single value output for the entire graph
        layer_name (str, optional): The name of the graph convolutional layer to use. Defaults to 'GCN1'.
        use_edge_attr (bool, optional): Whether to use edge attributes in the message passing layers. Defaults to True.
        edge_dim (int, optional): The dimensionality of the edge attributes. Defaults to 0.
        edge_attr_scaling (bool): Whether to scale the edge attributes to the same dimensionality as the input features. Defaults to False.
            Some PyTorch Geometric layers do not need to scale `edge_attr` through linear layers (e.g. NNConv), while others do.
            If enabled, `edge_attr` are always brought to the same last dimension as x, layer by layer.
        last_fully_adjacent (bool, optional): Whether to use a fully adjacent matrix in the final message passing layer. Defaults to False.
        skip_connect (bool, optional): Whether to use skip connections between message passing layers. Defaults to True.
        torch_seed (int or None, optional): The random seed to use for PyTorch. If None, no seed is set. Defaults to None.
        deterministic (bool, optional): Whether to use deterministic algorithms for PyTorch and PyTorch Geometric. Defaults to False.
        add_self_loops (bool, optional): Whether to add self-loops to the graph. Defaults to False.

    Attributes:
        task (str): The type of prediction task.
        dropout (float): The dropout probability to use across the model.
        num_layers (int): The number of message passing layers.
        skip_connect (bool): Whether to use skip connections between message passing layers.
        add_self_loops (bool): Whether to add self-loops to the graph.
        use_edge_attr (bool): Whether to use edge attributes in the message passing layers. These attributes are passed as `edge_weight`, `edge_type`, or `edge_attr`, 
            depending on the GNN layer.
        edge_dim (int): This variable serves for multiple purposes, depending on the context:
            - edge dimension to choose from `edge_attr` if the layer API only supports an `edge_weight` (negative indexing is permitted)
            - the input dimensionality of `edge_attr` linear layers, employed only if the layer API supports `edge_attr`.
        edge_attr_scaling (bool): Whether to scale the edge attributes to the same dimensionality as the input features.
            If enabled, `edge_attr` are always brought to the same last dimension as x, layer by layer.
            This will be set to False if `use_edge_attr` is False or `edge_dim` is 0.
        nonlin (callable): The nonlinearity to use across the model.
        layer_name (str): The name of the GNN layer to use.
        last_fully_adjacent (bool): Whether to use a fully adjacent matrix in the final message passing layer.
        transform_pre_mp (bool): Whether to apply a linear transformation to the input features before message passing.
        pre_mp_layer (nn.Module): The linear transformation to apply to the input features before message passing.
        layers (nn.ModuleList): The list of message passing layers.
        edge_layers (nn.ModuleList): The list of edge attribute layers.
        f_scorer (nn.Module): The module used to compute the final node scores.
        f_value (nn.Module): The module used to compute the final state value.
        h_prev (torch.Tensor): The node hidden states from the previous timestamp.
        deterministic (bool): Whether to use deterministic algorithms for PyTorch and PyTorch Geometric.
        detach_h (bool): Whether to detach the hidden state of the GNN layers from the computation graph.
        force_intern_h (bool): Whether to force the use of an internal hidden state for the GNN layers.
    """    
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, transform_pre_mp=False, num_layers=2, nonlin='ReLU', norm_layer=None, dropout=.2, dropout_head=0, 
                 task='node', layer_name='GCN1', use_edge_attr=True, edge_dim=0, edge_attr_scaling=False, last_fully_adjacent=False, skip_connect=True, torch_seed=None, 
                 deterministic=False, add_self_loops=False, **kwargs):
        super().__init__()
        self.deterministic = deterministic
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            torch.backends.cudnn.benchmark = False
            # this variable will force deterministic behavior w.r.t. both PyTorch and PyTorchGeometric
            if deterministic:
                torch.use_deterministic_algorithms(True)
            # use a 'silent' determinism setting otherwise (this does not cause errors if an op is non-deterministic)
            # but can lead to non-deterministic behavior in some cases (especially on CUDA)
            else:
                torch.use_deterministic_algorithms(False)
                torch.backends.cudnn.deterministic = True
        if dropout_head is None:
            dropout_head = dropout
                
        self.task = task
        self.dropout = dropout
        self.num_layers = num_layers
        self.skip_connect = skip_connect
        self.add_self_loops = add_self_loops
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim
        self.edge_attr_scaling = use_edge_attr and edge_dim > 0 and edge_attr_scaling
        self.layer_name = layer_name
        self.last_fully_adjacent = last_fully_adjacent

        # the nonlinearity to use across the model
        if nonlin:
            if nonlin.__contains__('Swish'):
                try:
                    Swish.beta = float(nonlin.split(':')[1])
                except (IndexError, ValueError):
                    pass
                self.nonlin = Swish
            else:
                self.nonlin = eval(f'nn.{nonlin}')
        else:
            self.nonlin = nn.Identity
        # variable to be used as argument for initializing special types of normalization layers that do not take an input dimensionality as argument
        # this will be populated by providing a string descriptor of this variable after ':' within the `norm_layer` argument
        norm_var = None
        if norm_layer:
            try:
                norm_layer, norm_var = norm_layer.split(':')
                norm_var = eval(norm_var)
            except ValueError:
                pass
            norm_layer = eval(f'pyg_nn.{norm_layer}')
        else:
            norm_layer = nn.Identity
        
        # Pre message-passing transformation layers initialization
        self.transform_pre_mp = transform_pre_mp
        if transform_pre_mp:
            self.pre_mp_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.nonlin()
            )
            input_dim = hidden_dim
        
        layers = []
        edge_layers = []

        ## Message-passing layers initialization
        in_channels, out_channels = input_dim, hidden_dim
        edge_in_channels = edge_dim
        for l_id in range(num_layers):
            if self.edge_attr_scaling:
                edge_layers.append(nn.Linear(edge_in_channels, in_channels))
            # some Conv layers do not project to a hidden_dim, therefore out_channels can often be the original in_channels
            gconv_layer, out_channels = self.build_conv_model(input_dim=in_channels, hidden_dim=out_channels, layer_id=l_id, **kwargs)
            layers += [
                gconv_layer,
                nn.Dropout(dropout),
                self.nonlin(),
                norm_layer(out_channels if norm_var is None else norm_var)
            ]
            # new edge_in_channels is the former in_channels
            edge_in_channels = in_channels
            # new in_channels is the former out_channels
            in_channels = out_channels

        # The entire suite of message-passing layers and edge attribute layers    
        self.hidden = nn.ModuleList(layers)
        if self.edge_attr_scaling:
            self.edge_hidden = nn.ModuleList(edge_layers)

        # By default, the head will either by a no-op or will project to `hidden_dim`, depending on the current `out_channels`
        # `out_channels` can either be `hidden_dim` or `input_dim` by this point, depending on the GNN layer chosen
        # Therefore, the default is to utilize the GNN as a backbone (no projection to a final `output_dim` is performed)
        self.head = None if out_channels == hidden_dim else nn.Linear(out_channels, hidden_dim)
        # If a `dropout_head` smaller than 1 is selected, however, the head will actually project to `output_dim` with a dropout set to `dropout_head`
        if dropout_head < 1:
            ## post message-passing
            self.head = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.Dropout(dropout_head), 
                # self.nonlin(), 
                nn.Linear(out_channels, output_dim))
        
    def build_conv_model(self, input_dim, hidden_dim, layer_id=0, **kwargs):
        """
        Builds a single convolutional layer for the GNN model. Given that Pytorch Geometric does not provide a unified API for all layers, this method attempts that.
        As such, this provides a highly customizable routine that can be used to build several types of GNN layer, with any extensions being easy to implement on top.

        Args:
            input_dim (int): The number of input features for each node in the graph.
            hidden_dim (int): The number of hidden features for each node in the graph after passing through this layer.
                Note, this may be ignored for certain layers, which may maintain the input dimensionality.
            layer_id (int, optional): The index of the current layer in the GNN model. Defaults to 0.
            **kwargs: Additional keyword arguments that can be used to customize the layer.

        Returns:
            tuple: A tuple containing the GNN layer object and the number of output features for each node.

        Notes:
            The following keyword arguments are supported:
                - `layer_args` (tuple): Additional arguments to pass to the GNN layer. Defaults to ().
                - `layer_kwargs` (dict): Additional keyword arguments to pass to the GNN layer. Defaults to {}.
        """
        # if last_fully_adjacent is True, instantiate a linear layer; combined with `global_max_pool`, this will effectively render an FA layer
        if self.last_fully_adjacent and layer_id == self.num_layers - 1:
            return nn.Linear(input_dim, hidden_dim), hidden_dim
        
        # first look for the layer type in GNN_LAYERS, dictionary of aliases that contains fully tested GNN layers
        try:
            layer_cls = GNN_LAYERS[self.layer_name]
        # the following is not fully tested for all layer possibilities, but we make an attempt to get the layer type by name from the namespace of `pyg_nn`
        except KeyError:
            layer_cls = getattr(pyg_nn, self.layer_name)
        
        conv_args = kwargs.get('layer_args', ())
        conv_kwargs = kwargs.get('layer_kwargs', {})
        
        if self.layer_name in ['GIN1', 'GINE']:
            args = (nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  self.nonlin(), nn.Linear(hidden_dim, hidden_dim)), )
            output_dim = hidden_dim
        elif self.layer_name in ['GAT1', 'GAT2'] and conv_kwargs.get('concat', True):
            args = (input_dim, hidden_dim // conv_kwargs.get('heads', 1))
            output_dim = hidden_dim
        elif self.layer_name == 'NNConv':
            edge_input_dim = self.edge_dim
            edge_output_dim = input_dim * hidden_dim
            args = (input_dim, hidden_dim, nn.Sequential(nn.Linear(edge_input_dim, edge_output_dim),
                                  self.nonlin(), nn.Linear(edge_output_dim, edge_output_dim)), )           
            output_dim = hidden_dim
        else:
            # check if the layer expects both `in_channels` and `out_channels`, in which case `output_dim` becomes `hidden_dim`
            if all(param in inspect.signature(layer_cls.__init__).parameters for param in ['in_channels', 'out_channels']):
                args = (input_dim, hidden_dim)
                output_dim = hidden_dim
            # otherwise, only the input channel is required, and the `output_dim` will remain the same as the `input_dim`
            else:
                args = (input_dim,)
                output_dim = input_dim
        # extend the current args list with elements obtained from `layer_args` entry in `**kwargs`  
        args += conv_args
        return layer_cls(*args, **conv_kwargs), output_dim

    def predict(self, *args, **kwargs):
        """
        Gets the true (i.e. first) output of the module for the given input.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The true output of the model.
        """
        return self.forward(*args, **kwargs)[0]

    def forward(self, x, edge_index, edge_attr=None, batch_idx=None, **kwargs):
        """
        Gets the full output of the module for the given input.

        Args:
            x (torch.Tensor): The input node features of shape (num_nodes, input_size).
            edge_index (torch.Tensor): The graph connectivity in COO format of shape (2, num_edges).
            edge_attr (torch.Tensor, optional): The edge features of shape (num_edges, edge_size). Defaults to None.
            batch_idx (torch.Tensor, optional): The batch index of each node of shape (num_nodes,). Defaults to None.
            **kwargs: Additional keyword arguments to pass to the message passing layers.

        Returns:
            tuple: The full output of the model. Contains: fully-transformed x and node embeddings.
        """
        if batch_idx is None:
            batch_idx = torch.zeros(len(x), dtype=int, device=x.device)
        if self.add_self_loops:
            edge_index, edge_attr = pyg_utils.add_self_loops(edge_index, edge_attr)
        # for fully deterministic behavior, we need to convert edge_index to a SparseTensor
        if self.deterministic and not isinstance(edge_index, SparseTensor):
            edge_index = SparseTensor(row=edge_index[1], col=edge_index[0], sparse_sizes=(len(x), len(x)))
            
        # transform input before MP, if option selected
        if self.transform_pre_mp:
            x = self.pre_mp_layer(x)
        # remember x before MP (may be an MLP representation already)
        x_0 = x
                
        # this includes both conv layers and 'extra' layers, such as nonlin and dropout
        total_hidden_layers = len(self.hidden)
        position_last_conv = int(total_hidden_layers * (1 - 1. / self.num_layers))
        # this will be used to index into edge_hidden
        conv_layer_counter = 0
        # note hidden contains both conv layers and nonlinear/norm layers
        for l_id, l in enumerate(self.hidden):
            # For graph layers, we need to add the `edge_index` tensor as an additional input.
            # All PyTorch Geometric graph layers inherit the class `MessagePassing`, hence we can simply check for this type.
            if isinstance(l, pyg_nn.MessagePassing):
                accepted_params = inspect.signature(l.forward).parameters
                
                final_params = [x]
                # allow for initial embedding to be supplied if the API allows it
                if 'x_0' in accepted_params:
                    final_params.append(x_0)
                final_params.append(edge_index)
                
                final_kwargs = {}
                if self.use_edge_attr and edge_attr is not None:
                    if 'edge_weight' in accepted_params:
                        if edge_attr.dim() == 2:
                            edge_attr = edge_attr[:, self.edge_dim]
                        final_kwargs = dict(edge_weight=edge_attr)
                    elif 'edge_type' in accepted_params:
                        if edge_attr.dim() == 2:
                            edge_attr = torch.argmax(edge_attr, dim=1)
                        final_kwargs = dict(edge_type=edge_attr)
                    elif 'edge_attr' in accepted_params:
                        if self.edge_attr_scaling:
                            edge_attr = self.edge_hidden[conv_layer_counter](edge_attr)
                        # The following can be commented out if one wants to add `edge_attr` to the SparseTensor representation
                        # if self.deterministic:
                        #     edge_index = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_attr, sparse_sizes=(len(x), len(x)))
                        final_kwargs = dict(edge_attr=edge_attr)
                
                x_conv = l(*final_params, **final_kwargs)
                conv_layer_counter += 1
                # we can add a skip connection if the option is enabled AND this is NOT the first layer (since that has a different `input_dim` dimensionality)
                # an exception happens when we have a `transform_pre_mp` selected, in which case the first layer's inputs also have `hidden_dim` dimensionality
                x = x + x_conv if self.skip_connect and (l_id or self.transform_pre_mp) else x_conv
                # keep track of the last embedding after message passing
                # for node/graph classification, we remove this quantity from the computation graph
                # for edge pred tasks, we need to be able to backprop through this
                if l_id == position_last_conv:
                    emb = x       
            else:
                # if last conv position encountered by a non-MP layer, then this is an FA layer
                if l_id == position_last_conv:
                    # average all features batch-wise, and feed through linear layer (equivalent to doing fully-adjacent layer)
                    # assert torch.allclose(global_mean_pool(x, batch_idx, deterministic=True).flatten(), global_mean_pool(x, batch_idx, deterministic=False))
                    global_x = l(global_mean_pool(x, batch_idx, deterministic=self.deterministic))
                    # add back to x, while keeping the same batch dimensionality (equivalent to skip connection with fully-adjacent layer)
                    x = emb = x + global_x[batch_idx]
                else:
                    # for non graph conv layers (nonlinearity/norm), simply pass through x
                    x = l(x)

        # for graph tasks, we perform a Readout operation
        if self.task == 'graph':
            x = global_mean_pool(x, batch_idx, deterministic=self.deterministic)
        # for edge tasks or when head is None, we return `x` without passing it through any head layers, together with the latent embedding post mesasge-passing    
        if self.head is None or self.task == 'edge':
            return x, emb
        # for both node and graph tasks, we pass `x` through the head layers, and return both the result and the embedding (the latter is disconnected from the torch graph)
        return self.head(x), emb.detach().cpu()
    
    
def global_mean_pool(x, batch_idx, deterministic=False):
    """
    Computes the global mean pooling of a batch of node features. Compared to the PyTorch Geometric implementation, this version supports a deterministic behavior.

    Args:
        x (torch.Tensor): The input node features of shape (num_nodes, num_features).
        batch_idx (torch.Tensor): The batch index of each node of shape (num_nodes,).
        deterministic (bool, optional): Whether to use a deterministic implementation of global mean pooling. Defaults to False.

    Returns:
        torch.Tensor: The global mean pooled node features of shape (batch_size, num_features).
    """
    if deterministic:
        M = torch.zeros(batch_idx.max()+1, len(x))
        M[batch_idx, torch.arange(len(x))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1).to(x.device)
        return M @ x
    else:
        return pyg_nn.global_mean_pool(x, batch_idx)
    
    
def init_weights(m):
    """
    Recursively initializes the weights of the model with a custom routine.

    Args:
        m (torch.nn.Module): The module to initialize the weights for.

    Returns:
        None
    """
    # also supports older versions of PYG which featured a `weight` parameter directly within the `MessagePassing` class
    if isinstance(m, nn.Linear) or isinstance(m, pyg_nn.MessagePassing) and hasattr(m, 'weight'):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    # # some modules may not have a bias or they may use the name with a bool value to trigger other behavior
    if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
        m.bias.data.fill_(0.01)