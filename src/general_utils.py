import math
import tqdm
import inspect
import numpy as np
from sys import stdout
from contextlib import contextmanager
from collections import deque


def round_decimals_up(number, decimals=2):
    """
    Returns a value rounded UP to a specific number of decimal places.

    Args:
        number (float): The number to be rounded up.
        decimals (int): The number of decimal places to round up to. Defaults to 2.

    Raises:
        TypeError: If decimals is not an integer.
        ValueError: If decimals is less than 0.

    Returns:
        float: The rounded up number.
    """
    if not isinstance(decimals, int):
        raise TypeError("The number of decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("The number of decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


### Workaround for TQDM to allow printing together with the progress line in the same STDOUT ###

def tqdm_print(*args, sep=' ', **kwargs):
    """
    Prints the given arguments to the console using tqdm's write method.

    Args:
        *args: The arguments to print.
        sep (str): The separator to use between arguments. Defaults to ' '.
        **kwargs: Additional keyword arguments to pass to tqdm's write method.
    """
    to_print = sep.join(str(arg) for arg in args)
    tqdm.tqdm.write(to_print, **kwargs)
        

@contextmanager
def redirect_to_tqdm():
    """
    Context manager to allow tqdm.write to replace the print function
    """   
    # Store builtin print
    old_print = print
    try:
        # Globaly replace print with tqdm.write
        inspect.builtins.print = tqdm_print
        yield
    finally:
        inspect.builtins.print = old_print

        
def tqdm_redirect(*args, **kwargs):
    """
    A generator function that redirects the output of tqdm progress bar to stdout.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Yields:
        x: The next value in the iterable.
    """
    with redirect_to_tqdm():
        for x in tqdm.tqdm(*args, file=stdout, **kwargs):
            yield x


def plot_dendrogram(model, counts, labels, **kwargs):
    """
    Plots a dendrogram using the given model, counts, and labels.

    Args:
        model (sklearn.cluster.AgglomerativeClustering): The model to use for clustering.
        counts (list): A list of counts for each node in the model.
        labels (list): A list of labels for each leaf node in the model.
        **kwargs: Additional keyword arguments to pass to the dendrogram function.

    Returns:
        The resulting dendrogram plot.
    """
    from scipy.cluster.hierarchy import dendrogram
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    n_samples = len(model.labels_)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, leaf_label_func=lambda child_idx: str(child_idx) if child_idx < n_samples else labels[child_idx - n_samples], 
                      distance_sort=True, **kwargs)


### Methods for constructing a PYG-compatible dataset from a record_agent instance ###

def get_dataset_mp(mp_result, sample_every=1, mark_delay_same_edges=False, name='EpidemicDataset', num_classes=2):
    """
    Constructs a `GeometricDataset` of graphs from a hierarchical representation of gathered information across simulations.

    Args:
        mp_result (numpy.ndarray): A hierarchical representation of gathered information across simulations.
            Its dimensions are num_iters x 3(i.e. xs, ys, edges) x num_events(i.e. time) x num_nodes x num_features.
        sample_every (int, optional): The sampling rate to use when constructing the dataset. Defaults to 1.
        mark_delay_same_edges (bool, optional): If the current set of edges does not differ from the previous timestamp's edges, this boolean determines
            whether the edges get duplicated in the resulting dataset, marked with a different time delay, or simply ignored. Defaults to False.
        name (str, optional): The name of the dataset. Defaults to 'EpidemicDataset'.
        num_classes (int, optional): The number of classes in the dataset. Defaults to 2.

    Returns:
        tuple: A tuple containing the constructed `GeometricDataset` of graphs and a list of the number of nodes in each entry.

    Notes:
        It is possible that the `mp_result` also holds entries that are not to be used for constructing the `GeometricDataset`.
        For convenience, the correct entries for the `GeometricDataset` will use integer keys (corresponding to their corresponding iteration number).
    """
    populate_dataset = GeometricDataset(name=name, num_node_features=mp_result[0][0][0].shape[1], num_classes=num_classes)
    lens = []
    for key, entry in mp_result.items():
        if isinstance(key, int):
            get_dataset(None, populate_dataset, sample_every, mark_delay_same_edges, *entry)
            lens.append(len(populate_dataset))
    return populate_dataset, lens


def get_dataset(record_agent=None, populate_dataset=None, sample_every=1, mark_delay_same_edges=False, xs=None, ys=None, edges_over_time=None):
    """
    Constructs a `GeometricDataset` from either a `record_agent` instance OR supplied values for features `xs`, labels `ys`, and links `edges_over_time`.

    Args:
        record_agent (RecordAgent, optional): A `RecordAgent` instance containing the data to be used for the dataset. 
            If not provided, `xs`, `ys` and `edges_over_time` must be supplied.
        populate_dataset (MutableSequence, optional): A `MutableSequence` equipped with an implementation for `append` that is populated with the constructed data. 
            If not provided, a new `GeometricDataset` will be created. 
            Note that `MutableSequences` like lists do not have the `num_node_features` and `num_classes` attributes, so may not be fully compatible with PyTorch Geometric.
        sample_every (int, optional): The sampling rate to use when constructing the dataset. Only every `sample_every`-th timestamp will be included in the dataset.
        mark_delay_same_edges (bool, optional): If the current set of edges does not differ from the previous timestamp's edges, this boolean determines
            whether the edges get duplicated in the resulting dataset, marked with a different time delay, or simply ignored. Defaults to False.
        xs (list[np.ndarray], optional): A list of numpy arrays containing the node features for each timestamp. 
            If not provided, the node features will be retrieved from the `record_agent` instance.
        ys (list[np.ndarray], optional): A list of numpy arrays containing the target labels for each timestamp. 
            If not provided, the target labels will be retrieved from the `record_agent` instance.
        edges_over_time (list[np.ndarray], optional): A list of numpy arrays containing the edges for each timestamp. 
            If not provided, the edges will be retrieved from the record_agent instance.

    Returns:
        GeometricDataset: A PyTorch Geometric-compatible dataset containing the constructed data.
    """
    # local imports to avoid globally importing `torch` in multiprocessing (which can lead to memory issues upon distribution)
    import torch
    from torch_geometric.data import Data
    # retrieve either from args or from `record_agent` instance
    xs = xs if xs else record_agent.xs
    ys = ys if ys else record_agent.ys
    if not edges_over_time:
        if record_agent.edges:
            edges_over_time = record_agent.edges
        else:
            raise ValueError('Edges need to be supplied, either as a parameter or from the recorded info')
    # verify whether all timestamped lists have the same amount of time indices        
    assert len(xs) == len(ys) == len(edges_over_time)
    # final dataset to return
    dataset = populate_dataset if populate_dataset is not None \
                else GeometricDataset(num_node_features=xs[0].shape[1], num_classes=2)                    
    edge_index = None
    edge_attr = None
    for t in range(0, len(xs), sample_every):
        try:
            ## establish edges for the graph at time t
            # if not None, transform to tensor, otherwise utilize edges_over_time from prev timestamp
            if edges_over_time[t] is not None:
                edges_current_time = torch.tensor(edges_over_time[t])
            # a dumy exception to escape the duplication of prev timestamp's edges, which otherwise would be accumulated further with different time delays,
            # for the case in which edges_over_time[t] is None (no new information available) AND mark_delay_same_edges is disabled
            elif not mark_delay_same_edges:
                raise SkipBlockException
            # transform list of edges into COO format for torch geometric
            edge_index_current = edges_current_time[:, :2].long().t().contiguous()
            # mark the time delay feature as 0 for the current timestamp
            edge_attr_current = torch.nn.functional.pad(input=edges_current_time[:, -1].float().reshape(-1, 1),
                                      pad=(0, 1, 0, 0), mode='constant', value=0)
            edge_index = torch.cat((edge_index, edge_index_current), dim=1)
            # increase by 1 the time delay of all other timestamps (Note, the current one is yet to be appended)
            # a clone is needed to avoid matching the timedelay increase in the other batches
            edge_attr = edge_attr.clone()
            edge_attr[:, 1] += 1
            edge_attr = torch.cat((edge_attr, edge_attr_current), dim=0)
        ## multiple exception causes are possible:
        # - edge_index is None, in which case we know this is the first timestamp, so no concat needs to happen
        except TypeError:
            edge_index = edge_index_current
            edge_attr = edge_attr_current
        # - edges_over_time[t] is None AND mark_delay_same_edges was disabled, in which case we want to utilize previous timestamp's edges with NO duplication
        except SkipBlockException:
            pass # edge_index and edge_attr in this case do not need other assignments
        dataset.append(Data(
            x=torch.from_numpy(xs[t]), 
            y=torch.tensor(ys[t], dtype=float),
            edge_index=edge_index, 
            edge_attr=edge_attr))
    return dataset


### Useful classes for RL and GNNs ###

class ReplayBuffer:
    """
    A replay buffer to be used for Monte Carlo reinforcement learning algorithms.

    Attributes:
        states (list): A list of states.
        actions (list): A list of actions.
        logp (list): A list of log probabilities.
        values (list): A list of values.
        rewards (deque): A deque of rewards.
        cndt_nodes (list): A list of candidate nodes.
        n_samples (int): The number of samples in the buffer.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logp = []
        self.values = []
        self.rewards = deque()
        self.cndt_nodes = []
        self.n_samples = 0
        
    def add(self, *entry):
        self.n_samples += 1
        for i, lst in enumerate(('states', 'actions', 'logp', 'values', 'rewards', 'cndt_nodes')):
            self.__dict__[lst].append(entry[i])
        
    def shift_and_discount_rewards(self, last_reward=0, gamma=.99, lamda=.97, reward_scale=1, norm_rwd=False, norm_adv=True):
        # rewards always correspond to the previous step, so they need to be shifted left
        self.rewards.popleft()
        # the last reward is supplied as an argument for convenience
        self.rewards.append(last_reward)
        # the last timestamp's value needs to be 0 as no further bootstrapping is possible
        # this will effectively make the last_reward to be the value of the last state for bootstrapping purposes
        self.values.append(0)
                
        lastgaelam = 0
        adv = np.zeros(self.n_samples)
        for i, r in enumerate(reversed(self.rewards)):
            # the actual timestamp is the reversed iterator
            t = self.n_samples - 1 - i
            delta = reward_scale * r + gamma * self.values[t+1] - self.values[t]
            adv[t] = lastgaelam = delta + gamma * lamda * lastgaelam
        # replace original rewards with normalized discounted rewards
        discounted_rewards = adv + np.array(self.values[:-1])
        self.rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7) \
                            if norm_rwd else discounted_rewards
        # replace original values with normalized GAE estimates
        self.values = (adv - adv.mean()) / (adv.std() + 1e-7) if norm_adv else adv

    def sample(self, batch_size=32, overlap=0):
        # a negative overlap will be converted to an overlap which ensures no incomplete batch
        if overlap < 0:
            overlap = self.n_samples % batch_size
        # values are discounted advantages, while rewards are discounted returns
        zipped = (self.states, self.actions, self.logp, self.values, self.rewards, self.cndt_nodes)
        for start in range(0, self.n_samples, batch_size - overlap):
            fin = start + batch_size
            if fin < self.n_samples:
                idx = range(start, fin, 1)
                done = False
            else:
                idx = range(start, self.n_samples, 1)
                done = True
            yield [list(map(lst.__getitem__, idx)) for lst in zipped]
            if done:
                break
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logp[:]
        del self.values[:]
        del self.rewards[:]


class GeometricDataset(list):
    """
    A list wrapper to be used as a dataset of graphs compatible with PyTorch Geometric.
    This extension is needed because some routines in PyTorch Geometric expect the attributes `num_node_features` and `num_classes`.

    Attributes:
        name (str): The name of the dataset.
        num_node_features (int): The number of features per node. If not provided, it is inferred from the first graph in the dataset.
        num_classes (int): The number of classes in the dataset.
    """
    def __init__(self, *args, name='EpidemicDataset', num_node_features=None, num_classes=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.num_node_features = num_node_features if num_node_features is not None else self[0].num_node_features
        self.num_classes = num_classes


class SkipBlockException(Exception):
    """
    Exception raised when a block inside a try segment is to be skipped.

    Notes:
        This exception does not represent an error per se.
        We use this to avoid duplicating edges when `mark_delay_same_edges` is disabled.
    """
    pass