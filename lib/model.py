import os
import inspect
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.functional as metrics

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
from torch_geometric.transforms import AddTrainValTestMask

import lib.rank_model as rankers
from lib.data import draw
from lib.model_utils import NoValidPB

from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from neptune.new.types import File

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, nonlin='ReLU', dropout=.2):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.nonlin = eval('nn.' + nonlin)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

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
        # Compute messages
        # x_i, x_j have shape [E, out_channels]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        # Update messages before return - can apply L2 norm here
        # aggr_out has shape [N, out_channels]
        aggr_out = F.normalize(aggr_out, p=2)
        return aggr_out

GNN_LAYERS = {
    "GCN": pyg_nn.GCNConv, # uses edge_weight
    "GraphConv": pyg_nn.GraphConv, # uses edge_weight
    "SAGE": pyg_nn.SAGEConv, # uses just a mean aggregator
    "GAT": pyg_nn.GATConv,
    "GIN": pyg_nn.GINConv, # as powerful as 1-WL, NO edge attr
    "GINE": pyg_nn.GINEConv, # uses edge_attr, same len as x
    "Custom": CustomConv, # can customize to use edge_attr
    "NNConv": pyg_nn.NNConv, # uses edge_attr of variable length
    "FAConv": pyg_nn.FAConv, # uses edge_weight in attention
    "RGCN": pyg_nn.RGCNConv, # uses edge_type
}

DENSE_GNN_LAYERS = {
    "D-GCN": pyg_nn.DenseGCNConv,
    "D-GraphConv": pyg_nn.DenseGraphConv,
    "D-SAGE": pyg_nn.DenseSAGEConv,
    "D-GIN": pyg_nn.DenseGINConv
}

CHECKPOINT_PATH = "saved"
DATASET_PATH = "data"


class MLPModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, nonlin='ReLU', dropout=0.1, **kwargs):
        """
        Inputs:
            input_dim - Dimension of input features
            hidden_dim - Dimension of hidden features
            output_dim - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dropout - Dropout rate to apply throughout the network
        """
        super().__init__()
        
        layers = []
        nonlin = eval("nn." + nonlin)
        
        # hidden layers
        in_channels, out_channels = input_dim, hidden_dim
        for l_id in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout),
                nonlin(inplace=True),
                nn.LayerNorm(out_channels)
            ]
            in_channels = out_channels    
        self.hidden = nn.Sequential(*layers)
            
        # final layer
        self.head = nn.Linear(in_channels, output_dim)
    
    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        x = self.hidden(x)
        return self.head(x), x


class GNNModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, transform_pre_mp=False, num_layers=3, nonlin='ReLU', norm_layer='LayerNorm', dropout=.2, dropout_head=0, task='node', layer_name='GCN', edge_dim=0, last_fully_adjacent=False, skip_connect=False, **kwargs):
        super().__init__()
        if not (task in ('node', 'edge', 'graph', 'timenode')):
            raise RuntimeError('Unknown task.')

        if dropout_head is None:
            dropout_head = dropout
                
        self.task = task
        self.dropout = dropout
        self.num_layers = num_layers
        self.skip_connect = skip_connect
        # this variable serves for multiple purposes, depending on the context:
        #  - edge dimension to choose from edge_attr if the layer API supports an edge_weight
        #  - the input dimensionality of edge_attr, to be used in the def of linear layers of edge_attr
        self.edge_dim = edge_dim
        # sometimes edge_attr do not need scaling through linear layers, so allow for this option to be disabled
        # if enabled, edge_attr are always brought to the same dim(-1) as x, layer by layer
        self.edge_attr_scaling = edge_dim > 0 and kwargs.get('edge_attr_scaling', False)
        # the nonlinearity to use across the model
        self.nonlin = eval(f'nn.{nonlin}') if nonlin else nn.Identity
        norm_layer = eval(f'nn.{norm_layer}') if norm_layer else nn.Identity
        # name of the graph conv layer
        self.layer_name = layer_name
        self.last_fully_adjacent = last_fully_adjacent
        
        # Allow transformations pre message-passing
        self.transform_pre_mp = transform_pre_mp
        if transform_pre_mp:
            self.pre_mp_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.nonlin(inplace=True)
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
                self.nonlin(inplace=True),
                norm_layer(out_channels)
            ]
            # new edge_in_channels is the former in_channels
            edge_in_channels = in_channels
            # new in_channels is the former out_channels
            in_channels = out_channels
            
        self.hidden = nn.ModuleList(layers)
        if self.edge_attr_scaling:
            self.edge_hidden = nn.ModuleList(edge_layers)

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
            try:
                dense_layer = DENSE_GNN_LAYERS[self.last_fully_adjacent]
                # kwargs supplied which can be used by the layer initialization
                dense_kwargs = kwargs.get('last_layer_kwargs', {})
                
                if self.last_fully_adjacent == 'D-GIN':
                    return dense_layer(nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                self.nonlin(inplace=True), 
                                nn.Linear(hidden_dim, hidden_dim)), **dense_kwargs), hidden_dim
                else:
                    return dense_layer(input_dim, hidden_dim, **dense_kwargs), hidden_dim
            except KeyError:
                # if no predef layer was found, but last_fully_adjacent is True, instantiate a linear layer (will be combined with global_max_pool to get FA layer)
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
                                  self.nonlin(inplace=True), nn.Linear(hidden_dim, hidden_dim)), )
            output_dim = hidden_dim
        elif self.layer_name == 'NNConv':
            edge_input_dim = self.edge_dim
            edge_output_dim = input_dim * hidden_dim
            args = (input_dim, hidden_dim, nn.Sequential(nn.Linear(edge_input_dim, edge_output_dim),
                                  self.nonlin(inplace=True), nn.Linear(edge_output_dim, edge_output_dim)), )           
            output_dim = hidden_dim
        else:
            # check if the layer expects both in_channels and out_channels, in which case output_dim becomes hidden_dim
            if all(param in inspect.signature(layer_cls.__init__).parameters for param in ['in_channels', 'out_channels']):
                args = (input_dim, hidden_dim)
                output_dim = hidden_dim
            # otherwise, only the input channel is required, and the output_dim will remain the same as the input_dim
            else:
                args = (input_dim, )
                output_dim = input_dim
        # extend the current args list with elements obtained from 'layer_args' entry in **kwargs        
        args += conv_args
        return layer_cls(*args, **conv_kwargs), output_dim

    def forward(self, x, edge_index, batch_idx=0, edge_attr=None, mask=None, **kwargs):
        # transform input before MP, if option selected
        if self.transform_pre_mp:
            x = self.pre_mp_layer(x)
        # remember x before MP (may be an MLP representation already)
        x_0 = x
                
        # based on self.last_fully_adjacent, allow for the last conv layer to be an Fully-Adjacent conv layer
        # if self.last_fully_adjacent designates a known DENSE_GNN_LAYER, make a fully connected dense adj matrix
        fully_connected_adj = None
        if self.last_fully_adjacent in DENSE_GNN_LAYERS:
            num_nodes = x.shape[0]
            # get a fully connected adj matrix (no self loops, self loops can be added via layer kwargs)
            fully_connected_adj = \
                torch.ones(num_nodes, num_nodes, device=edge_index.device) - \
                torch.eye(num_nodes, num_nodes, device=edge_index.device)
                
        # an iterator which gets incremeneted after each MessagePassing layer encountered
        conv_layer_count = 0 
        total_hidden_layers = len(self.hidden)
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
                if edge_attr is not None:
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
                            edge_attr = self.edge_hidden[conv_layer_count](edge_attr)
                        final_kwargs = dict(edge_attr=edge_attr)
                
                # result of MessagePassing layer
                x_conv = l(*final_params, **final_kwargs)
                conv_layer_count += 1
                # we can add a skip connection if the option is enabled AND this is NOT the first layer (since it has a different dimensionality)
                # an exception happens when we have a transform_pre_mp selected, in which case the first layer's inputs also have hidden_dim dimensionality
                x = x + x_conv if self.skip_connect and (l_id or self.transform_pre_mp) else x_conv
                # keep track of the last embedding after message passing
                # for node/graph classification, we remove this quantity from the computation graph
                # for edge pred tasks, we need to be able to backprop through this
                if conv_layer_count == self.num_layers:
                    emb = x.detach().cpu() if self.task != 'edge' else x        
            else:
                # a hack-ish way to check whether we're at the last convolution position, when the layer is fully adjacent (so not a subclass of MessagePassing)
                if l_id == total_hidden_layers - 4:
                    # by checking for this, we effectively know whether a predefined Dense layer was inputted, or a simple global_mean_pool + x is used
                    if fully_connected_adj is None:
                        # average all features batch-wise, and feed through linear layer (equivalent to doing fully-adjacent layer)
                        global_x = l(pyg_nn.global_mean_pool(x, batch_idx))
                        # add back to x, while keeping the same batch dimensionality (equivalent to skip connection with fully-adjacent layer)
                        x += global_x[batch_idx]
                    else:
                        # squeeze is needed since one dimension is added via the Dense layer (in this case we still have the batch dimensionality intact)
                        global_x = l(x, fully_connected_adj, mask=mask).squeeze(dim=0)
                        x = x + global_x if self.skip_connect else global_x
                    emb = x.detach().cpu() if self.task != 'edge' else x
                else:
                    # for non graph conv layers (nonlinearity/norm), simply pass through x
                    x = l(x)

        # for graph classification tasks, we need a readout op
        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch_idx)
        # for edge tasks, we return x without the head layers, and the embedding after mesasge-passing (still connected to torch graph)    
        if self.task == 'edge':
            return x, emb
        # for both node and graph tasks, we pass x through the head layers, and return both the result and the embedding (disconnected from torch graph)
        return self.head(x), emb
    

class GraphModel(pl.LightningModule):
    
    def __init__(self, model_name='GNN', classifier=True, lr=1e-3, weight_decay=1e-2, l1_reg=0, optim=None, use_edge_attr=False, inductive=False, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        self.lr = lr
        self.is_node_task = self.hparams.task == 'node'    
        self.is_edge_task = self.hparams.task == 'edge'
        # for node tasks, we can have an inductive setting in which nodes outside the split get masked out before MessagePassing
        self.hparams.inductive &= self.is_node_task
        # metrics to be computed based on whether this is an edge task OR node/graph classifier/regressor
        if self.is_edge_task:
            self.metric_keys = ('acc', 'f1', 'auc', 'ap')
            self.hparams.classifier = True
        # we denominate R2 as 'acc' in regression for convenience
        else:
            self.metric_keys = ('acc', 'recall', 'f1', 'auc', 'ap') if self.hparams.classifier else ('acc', 'mae', 'cos')
        
        if optim is not None:
            GraphModel.configure_optimizers = optim

        # for edge pred tasks, the model will be an encoder
        if hasattr(rankers, model_name):
            self.model = getattr(rankers, model_name)(**model_kwargs)
        elif model_name == 'MLP':
            self.model = MLPModel(**model_kwargs)
        elif model_name == 'GNN':
            self.model = GNNModel(**model_kwargs)
        # if model name is not MLP/GNN, we assume it designates a model which encloses our base GNN (e.g. GAE/VGAE with GNNModel as encoder)
        else:
            # in the case of VGAE, we need to output 2 quantities, so make the hidden_dim double
            if model_name == 'VGAE':
                model_kwargs['hidden_dim'] *= 2
                # we also set the beta_vae regularization coefficient
                self.beta_vae = model_kwargs.get('beta_vae', 1)
            self.model = GNNModel(**model_kwargs)
            self.enclosing_module = getattr(pyg_nn, model_name)(self.model)
            
        # allow custom losses to be supplied at runtime
        self.custom_loss = False
        if 'loss' in model_kwargs:
            self.loss_module = model_kwargs['loss']
            self.custom_loss = True
        # for edge prediction tasks, we need specialized losses (which should reside as an attr in the enclosing_module)
        elif self.is_edge_task:
            try:
                self.loss_module = self.enclosing_module.recon_loss
            except (AttributeError, ValueError):
                raise ValueError("For edge tasks, please also select an appropriate model_name, e.g. GAE/VGAE")
        # otherwise, choose appropriate loss based on whether this is a classifier/regressor and the output_dim
        else:
            if classifier:
                self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.output_dim == 1 else nn.CrossEntropyLoss()
            else:
                self.loss_module = nn.MSELoss()
        
    
    def forward(self, data, mode='train'):
        # batch - which graph this data belongs to (for multiple graph classification)
        x, batch_idx = data.x, data.batch
        # if no features, initialize to a vector of all 1's
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
            
        # the use_edge_attr boolean can be modified if the data object actually does not have any edge_attr
        use_edge_attr = self.hparams.use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None
        # decide which edge_index (sparse adjacency matrix) to use based on what type of problem we solve
        if self.is_edge_task:
            y = None
            edge_index = data.train_pos_edge_index
            edge_attr = data.train_pos_edge_attr if use_edge_attr else None
        # for node/graph level tasks, we use the whole sparse adjacency matrix
        else:
            y = data.y
            edge_index = data.edge_index
            edge_attr = data.edge_attr if use_edge_attr else None
            
        # prepare a mask to filter out the nodes outside the current split for the purpose of loss backprop and metrics compute
        if self.is_node_task:
            # Only calculate the loss on the nodes corresponding to the mask
            if mode == 'train':
                mask = data.train_mask
            elif mode == 'val':
                mask = data.val_mask
            elif mode == 'test':
                mask = data.test_mask
            else:
                assert False, 'Unknown forward mode: %s' % mode
            
        # perform message passing (this returns x and the embedding after the last convolution)
        x, emb = self.model(x, edge_index, batch_idx, edge_attr=edge_attr, mask=(mask if self.hparams.inductive else None))
        
        # depedning on the type of task, different types of losses and values for the 'acc' parameter will be returned
        if self.is_edge_task:

            # for VGAE, we split the output into mu and logstd
            if self.hparams.model_name == 'VGAE':
                middle = emb.shape[1] // 2
                mu = emb[:, :middle]
                logstd = emb[:, middle:]
                loss = self.loss_module(mu, edge_index) + self.beta_vae * self.enclosing_module.kl_loss(mu=mu, logstd=logstd)
            else:
                # the loss module is (by default) the reconstruction loss in the autoencoding case
                # this accepts as params the embedding and the train edge_index
                loss = self.loss_module(emb, edge_index)
                
            # in this setting, calculating train metrics doesn't make sense
            if mode == 'train':
                metric_vals = (1, 1, 1)
            else: 
                pos_edge_index = data[f'{mode}_pos_edge_index']
                neg_edge_index = data[f'{mode}_neg_edge_index']

                pos_y = emb.new_ones(pos_edge_index.size(1))
                neg_y = emb.new_zeros(neg_edge_index.size(1))
                y = torch.cat([pos_y, neg_y], dim=0).int()

                pos_pred = self.enclosing_module.decoder(emb, pos_edge_index, sigmoid=True)
                neg_pred = self.enclosing_module.decoder(emb, neg_edge_index, sigmoid=True)
                y_pred = torch.cat([pos_pred, neg_pred], dim=0)

                metric_vals = (metrics.accuracy(y_pred, y),
                               metrics.f1(y_pred, y),
                               metrics.auroc(y_pred, y), 
                               metrics.average_precision(y_pred, y))
        else:
            output_dim = x.shape[-1]
            # remove last redundant dimension (if any)
            x = x.squeeze(dim=-1)
            # mask out nodes out of the current split for loss computation (only for node tasks)
            if self.is_node_task:
                x, y = x[mask], y[mask]
            # calculate accuracy or r2 score, depending on 'classifier' parameter
            if self.hparams.classifier:
                # calculate the accuracy -> 2 cases: binary vs categorical
                if output_dim == 1:
                    num_classes = 2
                    # get predictions by thresholding
                    y_pred = (x > 0).int()
                    # convert y to expected float type by the loss
                    y = y.float()
                else:
                    num_classes = output_dim
                    # get argmax of x to infer y_pred
                    y_pred = x.argmax(dim=-1)
                    # convert y to expected long type by the loss
                    y = y.long()
                # calculate classification metrics #
                y_int = y.int() if y.dtype == torch.float else y
                metric_vals = (((y_pred == y).sum().float() / y.shape[0]),
                               metrics.recall(y_pred, y_int, num_classes=num_classes, average='macro'),
                               metrics.f1(y_pred, y_int, num_classes=num_classes, average='macro'),
                               metrics.auroc(x, y_int, num_classes=output_dim, average='macro'),
                               metrics.average_precision(x, y_int, num_classes=output_dim, average='macro'))
            else:
                # predictions are the unmodified x
                y_pred = x
                # convert y to expected float type by the loss 
                y = y.float()
                # calculate r^2 score
                metric_vals = (metrics.r2_score(y_pred, y), 
                               metrics.mean_absolute_error(y_pred, y), 
                               metrics.cosine_similarity(y_pred, y))
            
            if self.custom_loss:
                loss = self.loss_module(x, y, data)
            else:
                loss = self.loss_module(x, y)
            if self.hparams.l1_reg:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss += self.hparams.l1_reg * l1_norm
            
        return loss, dict(zip(self.metric_keys, metric_vals)), emb
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
#         optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2),
            'interval': 'step',
        }
        return [optimizer], [lr_scheduler]
        
        
    def training_step(self, batch, batch_idx):
        loss, metric_dict, emb = self.forward(batch, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_k, metric_v in metric_dict.items():
            self.log(f'train_{metric_k}', metric_v, on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        loss, metric_dict, _ = self.forward(batch, mode='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_k, metric_v in metric_dict.items():
            self.log(f'val_{metric_k}', metric_v, on_step=False, on_epoch=True, prog_bar=True)

        
    def test_step(self, batch, batch_idx):
        loss, metric_dict, _ = self.forward(batch, mode='test')
        for metric_k, metric_v in metric_dict.items():
            self.log(f'test_{metric_k}', metric_v, on_step=False, on_epoch=True, prog_bar=True)
        
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint['callback_metrics'] = self.trainer.callback_metrics
        
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

        
def train_classifier(dataset, **kwargs):
    return train_predictor(dataset, classifier=True, **kwargs)

def train_regressor(dataset, **kwargs):
    return train_predictor(dataset, classifier=False, **kwargs)

def train_predictor(dataset, classifier=True, model_name='GNN', layer_name='GCN', task='node', exp_id=None, test_splits=[.1, .1], loader_workers=0, ckp_monitor='val_acc', ckp_mode='max', ckp_best_k=3, ckp_patience=20, ckp_min_delta=1e-5, optim_func=None, seed=42, early_stop=False, tqdm_valid=False, eval_mode=True, trainer_ckp=None, logger=True, draw_node_size=None, lr=1e-3, batch_size=32, **trainer_model_kwargs):
    """
    Train a classifier with the supplied graph Dataset using PyTorch Lightning
    Currently, the seeding behavior in PL (and our code) works as follows:
     - seed = None: Use previous seed
     - seed >= 0: Set this seed
     - seed < 0: Set a new random seed, but with a warning (we circumvent this warning here)
    """
    # if a negative seed supplied, switch the seed to None to circumvent the warning
    if seed is not None and seed < 0:
        seed = None
        # note that we also need to pop this environment variable (if it exists) for PL to always reset the seed when 'seed'=None
        os.environ.pop("PL_GLOBAL_SEED", None)
        
    pl.seed_everything(seed)
    # behavior is deterministic when a seed >= 0 was supplied
    torch.backends.cudnn.determinstic = (seed is not None)
    # benchmarking (which can make the behavior nondeterministic) is disabled when a seed >= 0 was supplied
    torch.backends.cudnn.benchmark = (seed is None)
        
    # cleanup of the data may be needed if masks added dynamically at this point
    cleanup_needed = False
    # we assume only one graph in node classification / edge pred
    datapoint = dataset[0]
    
    # if a draw_node_size was supplied and the current logger is a NeptuneLogger, then we also log a drawing of the first datapoint
    if draw_node_size and isinstance(logger, NeptuneLogger):
        fig = draw(datapoint, plotly=True, layout='graphviz', figsize=(14, 14), 
                   degree_size=False, node_size=draw_node_size, seed=seed, show=False, legend=True,
                   labels_name=trainer_model_kwargs.get('labels_name', None))
        logger.experiment['visuals/graph'] = File.as_html(fig)
    
    if task == 'node':
        if not hasattr(datapoint, 'train_mask'):
            mask_generator = AddTrainValTestMask('train_rest', num_val=test_splits[0], num_test=test_splits[1])
            mask_generator(datapoint)
            cleanup_needed = True
        train_loader = val_loader = test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
    elif task == 'graph' or task == 'timenode':
        if test_splits[0] > 1:
            train_margin, val_margin = test_splits
        else:
            data_size = len(dataset)
            train_margin = int(data_size * (1 - sum(test_splits)))
            val_margin = int(data_size * (1 - test_splits[1]))
            
        train_loader = DataLoader(dataset[:train_margin], batch_size=batch_size, num_workers=loader_workers, shuffle=False)
        test_loader = DataLoader(dataset[val_margin:], batch_size=batch_size, num_workers=loader_workers, shuffle=False)
        # make the test set also a validation set if receiving equal/unsupported pairs of train-val margins
        # that is, val_margin will be moved to the end of the set
        if val_margin <= train_margin:
            val_margin = None
        val_loader = DataLoader(dataset[train_margin:val_margin], batch_size=batch_size, num_workers=loader_workers, shuffle=False)        
        
    elif task == 'edge':
        # since a locking mechanism exists for PYG datasets, datapoint will start being a local variable from this point
        pyg_utils.train_test_split_edges(datapoint, val_ratio=test_splits[0], test_ratio=test_splits[1])
        train_loader = val_loader = test_loader = DataLoader([datapoint], batch_size=1, shuffle=False)
        classifier = True
    else:
        assert False, 'Unknown training task: %s' % task
        
    # Define all callbacks
    callbacks = trainer_model_kwargs.get('callbacks', [])
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    callbacks.append(ModelCheckpoint(
        monitor=ckp_monitor,
        filename='sample-{epoch:02d}-{train_acc:.2f}-{' + ckp_monitor + ':.2f}',
        save_weights_only=False,
        save_top_k=ckp_best_k,
        mode=ckp_mode,
        verbose=False
    ))
    if early_stop:
        callbacks.append(EarlyStopping(
            monitor=ckp_monitor, 
            min_delta=ckp_min_delta, 
            patience=ckp_patience,
            mode=ckp_mode,
            verbose=False))
    if not tqdm_valid:
        callbacks.append(NoValidPB())
    
    # point to the root directory, according to the experiment, model name, and task type
    if not exp_id:
        try:
            exp_id = dataset.name.capitalize()
        except AttributeError:
            exp_id = 'Atest'
    task_name = '-'.join((exp_id, model_name + ('Classifier' if classifier else 'Regressor'), layer_name, task + 'Level'))
    root_dir = os.path.join(CHECKPOINT_PATH, task_name)
    os.makedirs(root_dir, exist_ok=True)
    
    # if selected lr is invalid (i.e. negative), conduct auto_lr_find and use abs(lr) as the min_lr
    auto_lr_find = (lr <= 0)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=20,
                         gpus=trainer_model_kwargs.get('gpus', '-1'),
                         max_epochs=trainer_model_kwargs.get('epochs', 20),
                         accelerator=trainer_model_kwargs.get('accelerator', None),
                         precision=trainer_model_kwargs.get('precision', 32),
                         auto_lr_find=auto_lr_find,
                         auto_scale_batch_size=None, # does not work with external dataloaders
                         logger=logger,
                         resume_from_checkpoint=trainer_ckp)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    trainer.logger._save_dir = root_dir
    
    # Check whether pretrained model exists. If yes, load it and skip training
    ckp_filename = os.path.join(CHECKPOINT_PATH, '%s.ckpt' % task_name)
    if os.path.isfile(ckp_filename):
        print('Found pretrained model, loading...')
    else:
        print('No pretrained found, training...')
        # if classifier and dataset.num_classes = 2, keep the output_dim as 1 (and use BCELoss); otherwise use CrossEntropyLoss
        # if NOT classifier, then use MSELoss
        model = GraphModel(model_name=model_name,
                           layer_name=layer_name,
                           classifier=classifier,
                           lr=abs(lr), # negative lr indicates tuning is turned on, but we want a positive value always
                           optim=optim_func,
                           input_dim=max(dataset.num_node_features, 1), # if no features, 1's will be added to x
                           output_dim=dataset.num_classes if classifier and dataset.num_classes > 2 else 1,
                           task=task,
                           batch_size=batch_size,
                           **trainer_model_kwargs)
        
        # tunes some hyperparameters before training (currently, batch_size and lr)
        if auto_lr_find:
            trainer.tune(model, train_dataloaders=train_loader,
                         lr_find_kwargs=dict(early_stop_threshold=5, mode='exponential', min_lr=abs(lr)),
                         scale_batch_size_kwargs=dict(init_val=batch_size))

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        ckp_filename = trainer.checkpoint_callback.best_model_path

        
    ### Evaluation
        
    # load best/pretrained Lightning model
    model = GraphModel.load_from_checkpoint(ckp_filename)
    # (re)log training metrics for the best model
    callback_metrics = torch.load(ckp_filename)['callback_metrics']
    trainer.logger.log_metrics({metric_k: metric_v for metric_k, metric_v in callback_metrics.items() if metric_k.startswith('train')})
    # validate model on validation set, and test on test set
    val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # evaluation metrics result dict
    result = {
        'train': callback_metrics['train_acc'].item(),
        'val': val_result[0]['val_acc'],
        'test': test_result[0]['test_acc']
    }
    
    
    ### Cleanup
    
    # if masks have been added dynamically, remove them
    if cleanup_needed:
        for attr in ('train_mask', 'val_mask', 'test_mask'):
            delattr(datapoint, attr)
    
    # If eval_mode=True, put the model in eval before returning (Note, this is needed because trainer.test switches the model back to training mode)
    if eval_mode:
        model.eval()
        
    return model, result, trainer


# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print("Train accuracy: %4.2f%%" % (100.0*result_dict["train"]))
    if "val" in result_dict:
        print("Val accuracy:   %4.2f%%" % (100.0*result_dict["val"]))
    print("Test accuracy:  %4.2f%%" % (100.0*result_dict["test"]))

# get best k models from Trainer
def get_best_k_models(trainer):
    models = []
    scores = []
    dct = trainer.checkpoint_callback.best_k_models
    for path, score in dct.items():
        models.append(GraphModel.load_from_checkpoint(path))
        scores.append(score)
    return models, scores
    