import inspect
import math
import itertools
import random
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_sparse import SparseTensor


GNN_LAYERS = {
    "GCN": pyg_nn.GCNConv, # uses edge_weight
    "GraphConv": pyg_nn.GraphConv, # uses edge_weight
    "SAGE": pyg_nn.SAGEConv, # uses just a mean aggregator
    "GAT": pyg_nn.GATConv, # latest version uses edge_attr for attention
    "GIN": pyg_nn.GINConv, # as powerful as 1-WL, NO edge attr
    "GINE": pyg_nn.GINEConv, # uses edge_attr, same len as x
    "NNConv": pyg_nn.NNConv, # uses edge_attr of variable length
    "FAConv": pyg_nn.FAConv, # uses edge_weight in attention
    "RGCN": pyg_nn.RGCNConv, # uses edge_type
    "GEN": pyg_nn.GENConv, # uses MessageNorm internally AND edge_attr
    "GAT2": pyg_nn.GATv2Conv,
}


class Model(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = 'cpu'

    def cpu(self, *args, **kwargs):
        on_device = super().cpu(*args, **kwargs)
        self.reset_device()
        return on_device
    
    def cuda(self, *args, **kwargs):
        on_device = super().cuda(*args, **kwargs)
        self.reset_device()
        return on_device
    
    def to(self, *args, **kwargs):
        on_device = super().to(*args, **kwargs)
        self.reset_device()
        return on_device
    
    def reset_device(self):
        raise NotImplementedError('This method needs to be overriden')
            

class EnsembleModel(Model):
    
    def __init__(self, input_dim, output_dim=1, n_models=3, combine='avg', torch_seed=None, **kwargs):
        super().__init__()
        base_seed = random.randint(0, 1e9) if torch_seed is None else torch_seed
        self.models = nn.ModuleList([RankingModel(input_dim, output_dim, torch_seed=base_seed + 3 * i_model, **kwargs) for i_model in range(n_models)])
        self.n_models = n_models
        self.combine = combine
        self.avg = combine.__contains__('avg')
        self.weight = nn.Parameter(torch.ones(2 * n_models).reshape(-1, 1)) if combine.__contains__('weight') else [1] * (2 * n_models)
        
    def __getitem__(self, i):
        return self.models[i]
    
    @property
    def h_prev(self):
        return self.models[0].h_prev
    
    @h_prev.setter
    def h_prev(self, value):
        for model in self.models:
            model.h_prev = value
            
    @property
    def deterministic(self):
        return self.models[0].deterministic
 
    def reset_device(self):
        self.device = self.models[0].nonlinear_scale.device 
        for model in self.models:
            model.device = self.device
        
        
    def predict(self, *args, i_model=-1, **kwargs):
        model = self.models[i_model] if i_model >= 0 else self
        return model.forward(*args, **kwargs)[0]
            
    def forward(self, *args, **kwargs):
        """
        The models within the ensemble may output only y_pred, only v_score or both depending on the 'scorer_type' argument
        """
        for i_model, model in enumerate(self.models):
            if i_model == 0:
                y_pred, v_score = model(*args, **kwargs)
            else:
                y_pred_i, v_score_i = model(*args, **kwargs)
                if y_pred_i is not None:
                    y_pred = y_pred + self.weight[i_model] * y_pred_i
                if v_score_i is not None:
                    v_score = v_score + self.weight[i_model + 1] * v_score_i
        
        if self.avg:
            y_pred = y_pred / self.n_models if y_pred is not None else None
            v_score = v_score / self.n_models if v_score is not None else None
            
        return y_pred, v_score


class RankingModel(Model):
    
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, rnn_layer=None, dropout_head=.1, agg='cat', state_score_method='max', last_incl_gnn_outs=False, reduce_h='', torch_seed=None, reset_h_seed=0, deterministic=False, **kwargs):
        super().__init__()
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
            # set a h_seed only if rest_h_seed is enabled
            if reset_h_seed > 0:
                self.h_seed = torch_seed  
    
        self.hidden_dim = hidden_dim
        self.last_incl_gnn_outs = last_incl_gnn_outs
        self.reduce_h = reduce_h
        # always mark task as timenode
        kwargs['task'] = 'timenode'
        # we completely 'dropout' the heads of the InfoModels by setting dropout_head = 1 (i.e. keep only backbones)
        self.info = InfoModel(input_dim, hidden_dim=hidden_dim, dropout_head=1, deterministic=deterministic, **kwargs)

        # REPLACE kwargs for diffusion model
        # the diffusion model has a single GNN layer as given by diffusion_layer_name
        kwargs['num_layers'] = 1
        kwargs['last_fully_adjacent'] = False
        # the diffusion model utilizes by default the first dimension of the edge attributes as edge weights (i.e. the actual edge weight of the network)
        # this behavior can be overwritten by passing as argument diff_edge_dim
        kwargs['edge_dim'] = 0
        kwargs['edge_attr_scaling'] = False
        for k, arg in kwargs.copy().items():
            if k.startswith('diff_'):
                key_to_replace = k[k.index('_')+1:]
                kwargs[key_to_replace] = arg
        self.diffusion = InfoModel(input_dim, hidden_dim=hidden_dim, dropout_head=1, deterministic=deterministic, **kwargs)
        self.nonlinear_scale = nn.Parameter(torch.tensor(1.), requires_grad=True)
        
        # hidden state of previous timestamp
        self.h_prev = None
        if reduce_h in ('GRU', 'LSTM'):
            g_hidden_model = RNN
            layers_head = kwargs.get('layers_head', 1)
            self.h_shape = [layers_head, 0, hidden_dim]
            self.mode = 1 if reduce_h == 'GRU' else 2
        else:
            g_hidden_model = MLP
            self.h_shape = [0, hidden_dim]
            self.mode = 0
        
        # CREATE MLP/RNN layers
        g_input_dim = f_input_dim = input_dim
        if agg == 'cat':
            g_input_dim += (3 if self.mode == 0 else 2) * hidden_dim
            # either 4 quantities (h, h_prev, d, i) or 2 quantities (h, h_prev) go into the final MLP
            f_input_dim += (4 if last_incl_gnn_outs else 2) * hidden_dim
        else:
            g_input_dim += hidden_dim
            f_input_dim += hidden_dim
        
        # the dropout of the MLP/RNN layers will be the supplied dropout_head, while the layers will be the layers_head
        self.g_hidden = g_hidden_model(g_input_dim, hidden_dim, agg=agg, rnn_layer=reduce_h, **kwargs)
        # last MLP will always have a linear head (to output_dim)
        kwargs['linear_head'] = True
        self.f_scorer = MLP(f_input_dim, output_dim, agg=agg, **kwargs)
        # the state-value MLP will have a state_scorer to output a value
        state_scorer = getattr(pyg_nn, f'global_{state_score_method}_pool')
        self.f_value = MLP(f_input_dim, output_dim, agg=agg, state_scorer=state_scorer, **kwargs)    
    
    def reset_device(self):
        self.device = self.nonlinear_scale.device
    
    
    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)[0]
        
    def forward(self, x, edge_index, edge_attr=None, edge_current=None, batch_idx=None, scorer_type=2, **kwargs):
        if batch_idx is None:
            batch_idx = torch.zeros(len(x), dtype=int, device=self.device)
        # lazily initialize h_prev if it wasn't yet initialized
        if self.h_prev is None:
            shape = self.h_shape
            # batch-size of h_prev comes at the 0th dim in the case of MLP but 1 for RNN (batch_first=False)
            # self.mode is 1/2 for RNN and 0 for MLP
            shape[1 if self.mode else 0] = x.shape[0]
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
                # we either always reset de seed (i.e. reset_h_seed = 2, full consistency) or reset just for evaluation (reset_h_seed = 1)
                if self.h_seed is not None and (self.reset_h_seed == 2 or not self.training):
                    torch.manual_seed(self.h_seed)
                self.h_prev = torch.randn(shape, device=self.device)
            if self.mode == 2:
                self.h_prev = (self.h_prev, torch.zeros_like(self.h_prev))
        if edge_current:
            edge_index_current, edge_attr_current = edge_current
        else:
            edge_index_current, edge_attr_current = edge_index, edge_attr
            
        d, _ = self.diffusion(x, edge_index_current, edge_attr_current, batch_idx, **kwargs)
        i, _ = self.info(x, edge_index, edge_attr, batch_idx, **kwargs)
        h = self.g_hidden(x, d, i, self.h_prev)
        
        if self.mode == 0:
            if self.reduce_h.__contains__('l2'):
                h = nn.functional.normalize(h, p=2., dim=-1)
            elif self.reduce_h.__contains__('tan'):
                h = torch.tanh(h)
            elif self.reduce_h.__contains__('sig'):
                h = torch.sigmoid(h)
            elif self.reduce_h.__contains__('mean'):
                h = (h - h.mean(0, keepdim=True))
            if self.reduce_h.__contains__('scl'):
                h = self.nonlinear_scale * h
            if self.reduce_h.__contains__('std'):
                h = h / h.std(0, unbiased=False, keepdim=True)
            h_prev = self.h_prev
            self.h_prev = h.detach()
        elif self.mode == 2:
            h_next, c_next = h
            h, h_prev = h_next[-1], self.h_prev[0][-1]
            self.h_prev = (h_next.detach(), c_next.detach())
        else:
            h_next = h.detach()
            h, h_prev = h[-1], self.h_prev[-1]
            self.h_prev = h_next
            
        inputs = [x, h, h_prev]
        if self.last_incl_gnn_outs:
            # d = nn.functional.normalize(d, p=2)
            # i = nn.functional.normalize(i, p=2)
            inputs += [d, i]
        # holder for node_scores and state_score
        output = [None, None]
        # for score_flag 0 or 2, we calculate a score for every node
        if scorer_type % 2 == 0:
            output[0] = self.f_scorer(*inputs).squeeze(dim=-1)
        # for score_flag 1 or 2, we calculate a score for the state
        if scorer_type >= 1:
            output[1] = self.f_value(*inputs, batch_idx=batch_idx).squeeze()
        
        return output
    
    
class RNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_head=.1, layers_head=1, agg='cat', rnn_layer='GRU', linear_head=False, **kwargs):
        super().__init__()
        self.agg = agg
        rnn_layer = eval(f'nn.{rnn_layer}')
        self.model = rnn_layer(input_dim, hidden_dim, num_layers=layers_head, dropout=dropout_head)
        self.output = nn.Linear(hidden_dim, output_dim) if linear_head else nn.Identity()
        
    def forward(self, *args, batch_idx=None):
        args, h_prev = args[:-1], args[-1]
        if self.agg == 'cat':
            inp = torch.cat(args, dim=1)
        else:
            inp = torch.zeros_like(args[1])
            for i, arg in enumerate(args[1:]):
                inp += arg
            inp = torch.cat((args[0], inp), dim=1)
        out, h = self.model(inp.unsqueeze(0), h_prev)
        # for LSTMS, h will be a tuple containing hidden state and c state
        if isinstance(h, tuple):
            return self.output(h[0]), h[1]
        return self.output(h)
    
    
class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_head=.1, layers_head=1, agg='cat', nonlin_head='ReLU', state_scorer=None, linear_head=True, **kwargs):
        super().__init__()
        self.agg = agg
        nonlin = nonlin_head if nonlin_head else kwargs.get('nonlin', 'Identity')
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


class InfoModel(nn.Module):
    
    def __init__(self, input_dim, output_dim=1, hidden_dim=64, transform_pre_mp=False, num_layers=3, nonlin='ReLU', norm_layer=None, dropout=.2, dropout_head=0, task='node', layer_name='GCN', use_edge_attr=True, edge_dim=0, last_fully_adjacent=False, skip_connect=True, torch_seed=None, deterministic=False, add_self_loops=False, **kwargs):
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
        # controls whether values are passed as edge_weight, edge_type, edge_attr, depending on the GNN layer
        self.use_edge_attr = use_edge_attr
        # this variable serves for multiple purposes, depending on the context:
        #  - edge dimension to choose from edge_attr if the layer API supports an edge_weight (SUPPORTS negative indexing)
        #  - the input dimensionality of edge_attr, to be used in the def of linear layers of edge_attr
        self.edge_dim = edge_dim
        # sometimes edge_attr do not need scaling through linear layers (e.g. NNConv), so allow for this option to be disabled
        # if enabled, edge_attr are always brought to the same dim(-1) as x, layer by layer
        self.edge_attr_scaling = use_edge_attr and edge_dim > 0 and kwargs.get('edge_attr_scaling', False)
        # the nonlinearity to use across the model
        self.nonlin = eval(f'nn.{nonlin}') if nonlin else nn.Identity
        # variable to be used for initializing special types of normalization layers
        # this will be populated by providing a string descriptor after ':' for the norm_layer variable
        self.norm_var = None
        if norm_layer:
            try:
                norm_layer, norm_var = norm_layer.split(':')
                self.norm_var = eval(norm_var)
            except ValueError:
                pass
            norm_layer = eval(f'pyg_nn.{norm_layer}')
        else:
            norm_layer = nn.Identity
        # name of the graph conv layer
        self.layer_name = layer_name
        self.last_fully_adjacent = last_fully_adjacent
        
        # Allow transformations pre message-passing
        self.transform_pre_mp = transform_pre_mp
        if transform_pre_mp:
            self.pre_mp_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.nonlin()
            )
            input_dim = hidden_dim
        
        layers = []
        edge_layers = []

        ## message-passing part
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
                norm_layer(out_channels if self.norm_var is None else self.norm_var)
            ]
            # new edge_in_channels is the former in_channels
            edge_in_channels = in_channels
            # new in_channels is the former out_channels
            in_channels = out_channels
            
        self.hidden = nn.ModuleList(layers)
        if self.edge_attr_scaling:
            self.edge_hidden = nn.ModuleList(edge_layers)

        # by default, the head will either by a no-op or will project to hidden_dim, depending on the current out_channels
        # out_channels can either be hidden_dim or input_dim, depending on the GNN layer chosen
        # thus, the default is to utilize the GNN as a backbone (no projection to the final output_dim)
        self.head = None if out_channels == hidden_dim else nn.Linear(out_channels, hidden_dim)
        # if a dropout_head smaller than 1 is selected, the head will actually project to output_dim
        if dropout_head < 1:
            ## post message-passing
            self.head = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.Dropout(dropout_head), 
                # self.nonlin(inplace=True), 
                nn.Linear(out_channels, output_dim))

        
    def build_conv_model(self, input_dim, hidden_dim, layer_id=0, **kwargs):
        # if this is the last conv layer, we may want to have additional functionality: e.g. special Dense layer
        # the dense layyer will be added ONLY IF it is a recognised Dense layer (i.e. part of DENSE_GNN_LAYERS)
        if self.last_fully_adjacent and layer_id == self.num_layers - 1:
            # if last_fully_adjacent is True, instantiate a linear layer; combined with global_max_pool renders FA layer
            return nn.Linear(input_dim, hidden_dim), hidden_dim
        
        # first look for the layer type in GNN_LAYERS (dict of aliases)
        try:
            layer_cls = GNN_LAYERS[self.layer_name]
        # even if not fully tested, make an attempt to initialize the layer by name from the namespace of pyg_nn
        except KeyError:
            layer_cls = getattr(pyg_nn, self.layer_name)
        
        # args and kwargs which can be used by the layer init
        conv_args = kwargs.get('layer_args', ())
        conv_kwargs = kwargs.get('layer_kwargs', {})
        
        if self.layer_name in ['GIN', 'GINE']:
            args = (nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  self.nonlin(), nn.Linear(hidden_dim, hidden_dim)), )
            output_dim = hidden_dim
        elif self.layer_name == 'NNConv':
            edge_input_dim = self.edge_dim
            edge_output_dim = input_dim * hidden_dim
            args = (input_dim, hidden_dim, nn.Sequential(nn.Linear(edge_input_dim, edge_output_dim),
                                  self.nonlin(), nn.Linear(edge_output_dim, edge_output_dim)), )           
            output_dim = hidden_dim
        else:
            # check if the layer expects both in_channels and out_channels, in which case output_dim becomes hidden_dim
            if all(param in inspect.signature(layer_cls.__init__).parameters for param in ['in_channels', 'out_channels']):
                args = (input_dim, hidden_dim)
                output_dim = hidden_dim
            # otherwise, only the input channel is required, and the output_dim will remain the same as the input_dim
            else:
                args = (input_dim,)
                output_dim = input_dim
        # extend the current args list with elements obtained from 'layer_args' entry in **kwargs        
        args += conv_args
        return layer_cls(*args, **conv_kwargs), output_dim


    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)[0]

    def forward(self, x, edge_index, edge_attr=None, batch_idx=None, **kwargs):
        if batch_idx is None:
            batch_idx = torch.zeros(len(x), dtype=int, device=x.device)
        if self.add_self_loops:
            edge_index, edge_attr = pyg_utils.add_self_loops(edge_index, edge_attr)
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
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layers inherit the class "MessagePassing", hence
            # we can simply check the class type.
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
                            # edge_index = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_attr, sparse_sizes=(len(x), len(x)))
                        final_kwargs = dict(edge_attr=edge_attr)
                
                x_conv = l(*final_params, **final_kwargs)
                conv_layer_counter += 1
                # we can add a skip connection if the option is enabled AND this is NOT the first layer (since it has a different dimensionality)
                # an exception happens when we have a transform_pre_mp selected, in which case the first layer's inputs also have hidden_dim dimensionality
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
                    x += global_x[batch_idx]
                    emb = x
                else:
                    # for non graph conv layers (nonlinearity/norm), simply pass through x
                    x = l(x)

        # for graph classification tasks, we need a readout op
        if self.task == 'graph':
            x = global_mean_pool(x, batch_idx, deterministic=self.deterministic)
        # for edge tasks, we return x without the head layers, and the embedding after mesasge-passing (still connected to torch graph)    
        if self.head is None or self.task == 'edge':
            return x, emb
        # for both node and graph tasks, we pass x through the head layers, and return both the result and the embedding (disconnected from torch graph)
        return self.head(x), emb.detach().cpu()
    
    
def global_mean_pool(x, batch_idx, deterministic=False):
    # Note: pyg_nn.global_mean_pool() is NOT deterministic, so an alternative that is needs to be in-place for 'deterministic' to be enabled
    if deterministic:
        M = torch.zeros(batch_idx.max()+1, len(x))
        M[batch_idx, torch.arange(len(x))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1).to(x.device)
        return M @ x
    else:
        return pyg_nn.global_mean_pool(x, batch_idx)