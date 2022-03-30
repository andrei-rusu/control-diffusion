import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import rgb2hex
import pandas as pd
import networkx as nx

import torch
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

class GeometricDataset(list):
    def __init__(self, *args, name=None, num_node_features=None, num_classes=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.num_node_features = num_node_features if num_node_features is not None else self[0].num_node_features
        self.num_classes = num_classes

def get_dataset(name="cora", normalize=True, use_node_attr=True, use_edge_attr=True, cleaned=False, largest_component=False, use_gdc=-1):
    dataset = None
    name_lwr = name.lower()
    path = "data/" + name_lwr
    
    transform = T.Compose([T.NormalizeFeatures() if normalize else T.Constant(value=0)])
    
    if name_lwr in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=path, name=name_lwr, transform=transform)
    else:
        try:
            dataset = TUDataset(root=path, name=name, use_node_attr=use_node_attr, 
                                use_edge_attr=use_edge_attr, cleaned=cleaned,
                                transform=transform)
        except:
            raise ValueError("Dataset not supported")
            
    # we may want to work just with the maximum connected component
    if largest_component:
        # we will return a GeometricDataset since Planetoid/TUDataset cannot be modified in-place
        new_dataset = GeometricDataset(name=name, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes)
        for datapoint in dataset:
            # establish which attributes to copy over to the networkx transform
            attrs = filter(lambda attr: hasattr(datapoint, attr), ['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
            # convert PYG Data to networkx
            net = to_networkx(
                    datapoint, 
                    node_attrs=attrs,
                    edge_attrs=['edge_attr'] if datapoint.edge_attr is not None else [])
            # select only the largest connected component and convert back to a PYG Data object
            component_data = from_networkx(net.subgraph(max(nx.weakly_connected_components(net), key=len)))
            # populate a new dataset with the new datapoint containing just the largest component
            new_dataset.append(component_data)
        dataset = new_dataset
    
    # use_gdc is disabled if -1
    # if 0, exact is False, if 1 exact is True
    if use_gdc != -1:
        # we will return a GeometricDataset since Planetoid/TUDataset cannot be modified in-place
        new_dataset = GeometricDataset(name=name, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes)
        # the GDC transform
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05, eps=0.05),
                    sparsification_kwargs=dict(method='threshold', avg_degree=64), 
                    exact=bool(use_gdc))
        for datapoint in dataset:
            new_dataset.append(gdc(datapoint))
        dataset = new_dataset
    
    return dataset

def get_dataset_from_network(G, x=None, y=None, edge_attr=None, num_classes=None, largest_component=False, use_gdc=-1, name=None):
    # we may want to work just with the maximum connected component
    if largest_component:
        connected_components = nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G)
        G = G.subgraph(max(connected_components, key=len))
    # we select the indexes in the active component to filter x,y,edge_attr
    indexes_in_component = list(G)
    # convert to Data PYG object
    data = from_networkx(G)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if x is None:
        data.x = torch.ones(len(G), 1)
    else:
        data.x = torch.tensor(x[indexes_in_component])
    if y is not None:
        data.y = torch.tensor(y[indexes_in_component], dtype=torch.long)
        # if no num_classes provided, default it to the number of distinct elements in y
        if num_classes is None:
            num_classes = len(set(data.y))
    else:
        # if no y supplied and no num_classes entered, default num_classes to 2
        if num_classes is None:
            num_classes = 2
    if edge_attr is not None:
        data.edge_attr = edge_attr[indexes_in_component]
    
    # use_gdc is disabled if -1
    # if 0, exact is False, if 1 exact is True
    if use_gdc != -1:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05, eps=0.05),
                    sparsification_kwargs=dict(method='threshold', avg_degree=64), 
                    exact=bool(use_gdc))
        data = gdc(data)

    dataset = GeometricDataset([data], name=name, num_classes=num_classes)
    return dataset    

def get_random_dataset(model='erdos_renyi_graph', n=30, x=None, y=None, edge_attr=None, largest_component=False, use_gdc=-1, **model_kwargs):
    G = getattr(nx, model)(n, **model_kwargs)
    name = model.split('_')[0]
    return get_dataset_from_network(G, x, y, edge_attr, largest_component=largest_component, use_gdc=use_gdc, name=name)

def get_dataset_from_adj(adjmatrix, x=None, y=None, edge_attr=None, largest_component=False, use_gdc=-1, name=None, **model_kwargs):
    adj = np.nan_to_num(adjmatrix)
    G = nx.convert_matrix.from_numpy_matrix(adj)
    return get_dataset_from_network(G, x, y, edge_attr, largest_component=largest_component, use_gdc=use_gdc, name=name)


def draw(G, plotly=False, labels=True, labels_name=None, layout_type="spring", degree_size=False, node_size=20, figsize=None, legend=True, show=True, with_ids=False, **layout_kwargs):
    """
    Draw a PYG Data / nx Graph object using networkx/plotly libraries
    """    
    graph_to_draw = G
    if isinstance(G, Data):
        graph_to_draw = to_networkx(G, to_undirected=G.is_undirected())
        # an empty labels list can get populated at this stage from G.y
        # Note, labels = None completely disables collecting any labels from G
        if labels is True:
            try:
                labels = G.y.detach().numpy()
            except AttributeError:
                pass
        
    labels_set = None
    try:
        labels_set = sorted(set(labels))
        # Set a color map to distinguish between classes
        cmap = plt.get_cmap('Set2')
        color_map = []
        for i, node in enumerate(graph_to_draw):
            color_map.append(cmap(labels[i]))
    except TypeError:
        color_map = 'grey'
    
    if labels_name is not None:
        legend_label = lambda label: labels_name[label]
        plt.subplots_adjust(left=.2)
    else:
        legend_label = lambda label: str(label)
        plt.subplots_adjust(left=.1)  
            
    # generate a drawing layout for the current network        
    pos = generate_layout(graph_to_draw, layout_type, **layout_kwargs)
    # recover degrees for every node
    degrees = nx.degree(graph_to_draw)
    degrees_values = list(dict(degrees).values())
    # if degree_size selected, we also ammend the node_size of each node to be according to its degree
    if degree_size:
        amplifier = 5
        node_size = np.array([(degrees[node]+1) * amplifier for node in graph_to_draw.nodes])
        
    if not plotly:
        node_size_scale_for_matplotlib = 20  # node_size scaled up for matplotlib compared to plotly
        fig = plt.figure(figsize=figsize)
        nx.draw(graph_to_draw, pos=pos, node_color=color_map, with_labels=with_ids, edgecolors='#888',
                edge_color='black', node_size=node_size * node_size_scale_for_matplotlib, font_weight='bold')
        if legend:
            # create color legend from the labels_set
            plt.legend(handles=[mpatches.Patch(color=cmap(label), label=legend_label(label)) for label in labels_set],
                       loc='upper left', prop={'size': 12}, bbox_to_anchor=(0, 1), bbox_transform=plt.gcf().transFigure)       
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig

    else:
        import plotly.graph_objects as go

        inch_to_px = plt.rcParams['figure.dpi']  # inches to pixels
        plotly_fig = go.Figure(
             layout=go.Layout(
                autosize=False,
                width=figsize[0] * inch_to_px,
                height=figsize[1] * inch_to_px,
                showlegend=legend,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest',
                margin=dict(b=10, l=5, r=5, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
        
        # Add edges to figure
        edge_x = []
        edge_y = []
        for edge in graph_to_draw.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        plotly_fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        ))
        
        # Add nodes to figure
        node_x = []
        node_y = []
        for node in graph_to_draw.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if with_ids:
                plotly_fig.add_annotation({
                    'align': 'center',
                    'font': {'family': 'Arial','color': 'black', 'size': node_size / 2},
                    'opacity': 1,
                    'showarrow': False,
                    'text': f'<b>{node}</b>',
                    'x': x,
                    'xanchor': 'center',
                    'xref': 'x',
                    'y': y,
                    'yanchor': 'middle',
                    'yref': 'y'
                })

        # separate logic for when nodes have labels and when they don't    
        if labels_set is None:
            plotly_fig.add_trace(go.Scatter(
                name='Nodes',
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    color=color_map,
                    size=node_size,
                    line_width=2),
                text=[f'Node: {node}<br />Class: Unknown<br />Connections: {degree}' \
                      for node, degree in zip(graph_to_draw.nodes, degrees_values)]
            ))
        else:
            # we need to convert these lists to numpy arrays in order to select multiple indexes
            nodes = np.array(graph_to_draw.nodes)
            node_x = np.array(node_x)
            node_y = np.array(node_y)
            degrees_values = np.array(degrees_values)
            for label in labels_set:
                # look in the 'labels' collection for the indexes where the current 'label' can be found
                indexes = np.where(labels == label)
                colors = cmap(label, bytes=True)
                plotly_fig.add_trace(go.Scatter(
                    name=legend_label(label),
                    x=node_x[indexes], y=node_y[indexes],
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                        color=f'rgb({colors[0]},{colors[1]},{colors[2]})',
                        size= np.array(node_size)[indexes] if degree_size else node_size,
                        line_width=2),
                    text=[f'Node: {node}<br />Class: {legend_label(label)}<br />Connections: {degree}' \
                          for node, degree in zip(nodes[indexes], degrees_values[indexes])]
                ))

        return plotly_fig
    
    
def generate_layout(G, layout_type='spring_layout', **kwargs):
    # deals with the case in which the layout was not specified with the correct suffix
    if not layout_type.endswith('_layout'):
        layout_type += '_layout'
    try:
        # get the method from networkx which generates the selected layout
        method_to_call = getattr(nx, layout_type)
    except AttributeError:
        # graphviz layouts will be captured by this block
        method_to_call = getattr(nx.nx_agraph, layout_type)
    # get signature of the method to make sure we pass in only the supported kwargs per each layout type
    signature = inspect.signature(method_to_call)
    # skip positional, *args and **kwargs arguments
    skip_kinds = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    # loop through the method signature parameters and either put the value supplied by kwargs or retain the default value
    passed_kwargs = {
        param.name: kwargs[param.name]
        for param in signature.parameters.values()
        if param.name in kwargs and param.kind not in skip_kinds
    }
    # generate drawing layout as selected
    return method_to_call(G, **passed_kwargs)
