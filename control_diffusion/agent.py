import os
import warnings
import json
import copy
import heapq
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from time import sleep
from pydoc import locate
from contextlib import nullcontext
from collections import deque, Counter
from collections.abc import Iterable

from .general_utils import tqdm_redirect, round_decimals_up, plot_dendrogram, ReplayBuffer


UNACCEPTED_FOR_TEST_TRACE = {'H', 'D'}
PERCEIVED_UNINF_STATES = {'S', 'E', 'R'}
CKP_LOCATION = 'models/'
TB_LOCATION = 'tensorboard/'


class Agent:
    """
    A class representing a control agent for network-based diffusion processes.

    Attributes:
        tester (function): The function to use for ranking nodes for testing.
        tracer (function): The function to use for ranking nodes for testing.
        vaxer (function): The function to use for ranking nodes for vaccination.
        n_nodes (int): The number of nodes in the network.
        n_test (int): The number of nodes to test at each control iteration.
        n_trace (int): The number of nodes to trace at each control iteration.
        n_vax (int): The number of nodes to vaccinate at each control iteration.
        all_nodes (list): A list of all node IDs in the network.
        set_infected_attr (bool): Whether to set the `node_infected` attribute on the network object based on its `node_states` attribute. 
            If in any iteration the network object does not have this attribute, this is set to True for all subsequent iterations.
        keep_pos_until (int): The number of control iterations during which the agent keeps positive test results.
        keep_neg_until (int): The number of control iterations during which the agent keeps negative test results.
        pos_added (deque): A queue of the number of positives added at each control iteration.
        neg_added (deque): A queue of the number of negatives added at each control iteration.
        pos (deque): A queue of the positive node IDs that have been tested recently. This gets dequeued after `keep_pos_until` control iterations.
        neg (deque): A queue of the negatives node IDs that have been tested recently. This gets dequeued after `keep_neg_until` control iterations.
        see_all_uninfected (bool): Whether to recognize upon testing all uninfected nodes, according to the Network object (True), 
            or only the "perceived" ones, according to `PERCEIVED_UNINF_STATES` (False).
        recollection (float): Parameter which controls the type of test recollection and how it is used.
            - 0: recollection of positive results, network to sample from remains unchanged.
            - 1: recollection of positive and negative results, network to sample from remains unchanged.
            - >=2: recollection of positive and negative results, network to sample from is formed only of active nodes (i.e. not isolated).
            - >=3: recollection of positive and negative results, network to sample from is formed only of active nodes that have not been tested negative.
            Special setting: if not an integer, centrality features are forced to update at each iteration according to the network to sample from.
        k_hops (int): The number of hops to consider when computing infectious neighborhood node features.
        norm_hops (float): The normalization factor for the k-hops node features. This is either 1/(n_nodes - 1) if k_hops < 0, or 1 (i.e. ignored) otherwise.
        known_inf_neigh (np.ndarray): The k-hops node features, consisting of the sum of the normalized infectious neighbors at each depth k.
        computed_measures (list): A list of computed node ranking measures. These will be computed according to the agent's ranking policy.
        prev_test_feat (bool): Whether to use previous test results as features.
        dynamic_feat (np.ndarray): The dynamic node features, consisting of the (prev_untested, prev_pos, prev_neg) one-hot vector and the boolean ever_pos. 
            These features are updated at each control iteration.
        net_changed (bool): Whether the network has changed since the last control iteration. This is used to avoid recomputing node ranking measures when not needed.
        inf_p (np.ndarray): The probability of infection for each node in the network. Can be obtained from an SL agent or from the infection likelihood 
            observed in a node's cluster.
        seed (int): The random seed to use for the agent sampling.
        episode (int): The current episode number.
        eps (float): The episod-specific epsilon value for the agent.
        sim_id (int): The ID of the simulation.
        control_iter (int): The current control iteration.
        control_day (int): The current control day.
        mix_learner (bool): Whether the agent is a learning agent in a MixAgent that can combine different strategies for each control operation.
        trace_start (int): Iteration when the trace logic starts;
        trace_latency (int): Days to wait until tracing/isolation takes effect after the node has been chosen for tracing
        trace_ignore (float): Whether to consider all neighbors of positives for tracing or just a subset of them. This can be:
            - 0: all neighbors in an "active" state considered
            - 1: newly-identified positives are excluded
            - 1.5: newly-tested nodes are excluded
            - 2: new positives and negative history are excluded
        trace_till (int): Maximum budget allowed for tracing throughout the simulation.
        vax_start (int), vax_latency (int), vax_till (int): Mirroring the above functionality, but for vaccination rather than tracing.
        vax_ignore (float): Whether to consider all unvaccinated nodes for vaccination or just a subset of the nodes. This can be:
            - 0: all non-orphan nodes that are unvaccinated are considered
            - 1: only unvaccinated nodes in an "active" state considered
            - 1.5: only unvaccinated nodes that were deemed valid for testing prioritization are considered (i.e. excluding recent negatives)
            - 2: only unvaccinated nodes in an "active" state that were NOT tested positive or traced in the current iteration are considered
            - 2.5: only unvaccinated nodes that were deemed valid for testing prioritization and were NOT tested positive or traced in the 
                current iteration are considered.
        valid_for_trace (np.ndarray): An array of node IDs that were valid for tracing in the last step there were more such nodes than the daily budget `n_trace`. 
            This is used to perform a mock forward pass in learning-based tracing agents to avoid hidden state drifts when `n_trace` > `valid_for_trace` (current).
        valid_for_vax (list): A list of node IDs that were valid for vaccination in the last step there were more such nodes than the daily budget `n_vax`. 
            This is used to perform a mock forward pass in learning-based vaccination agents to avoid hidden state drifts when `n_trace` > `valid_for_vax` (current).
        cluster_args (tuple): Additional arguments to pass to the clustering function (incl. cluster_args[0] which controls display of dendrogram).
        scorer_type (int): The type of scorer used by the learning agent's ranking model. This can be:
            - 0: only node scorer used (e.g. for SL agents)
            - 1: only state scorer used (e.g. for RL critics that do not function as RL actors as well)
            - 2: node scorer + state scorer used (e.g. for RL agents).
        explaining (tuple): A tuple of booleans indicating whether to display explanations for testing and tracing.
        display (function): The function to use for displaying explanations.
        ax (matplotlib.axes.Axes): The axes object used for displaying explanations.
        save_path (str): The path to save the control logging information.
        name (str): The name of the agent that will be used for logging.
        debug_print (int): The increment of control iterations after which to print debug information. If 0, no debug information is printed.
        ckp_ep (int): The number of control episodes after which to save a checkpoint.
        ckp (int): The number of control iterations after which to save a checkpoint.
        tb_log (bool): Whether to log data on TensorBoard. If True, a SummaryWriter object is created and information is logged on TensorBoard
            If False, no SummaryWriter object is created, and therefore the overhead of importing `torch` can be circumvented.
        tb_layout (dict): The layout of the TensorBoard log, which can be used to log multiple traces on the same graph.
        writer (torch.utils.tensorboard.SummaryWriter): The SummaryWriter object used for logging on TensorBoard.
        loss_log (float): The loss value to log on TensorBoard. This will be different than 0 only for learning agents.

    """
    @staticmethod
    def from_dict(typ='centrality', **kwargs):
        """
        Create an agent object from a dictionary of parameters.

        Args:
            typ (str): The name of the agent to create. The convention is to use the agent type followed by a dash and then the agent identifier.
            **kwargs: Additional keyword arguments to pass to the agent constructor.

        Returns:
            An instance of the specified agent type.
        """
        return AGENT_TYPE[typ.split('-')[0].lower()](name=typ.capitalize(), **kwargs)
    
    @staticmethod
    def model_from_dict(**kwargs):
        """
        Create a Model object from a dictionary of parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            An instance of the specified Model type.
        """
        from .rank_model import Model
        return Model.from_dict(**kwargs)
           
    @classmethod
    def get_subclasses(cls):
        """
        Recursively get all subclasses of the class.

        Args:
            cls (type): The class to get the subclasses of.

        Yields:
            type: The next subclass of the class.
        """
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass
        
    def __init__(self, tester=True, tracer=False, vaxer=False, n_nodes=200, n_test=2, n_trace=0, n_vax=0, keep_pos_until=6, keep_neg_until=3, recollection=1, k_hops=0, 
                 name=None, seed=None, see_all_uninfected=True, debug_print=0, episode=0, epsilon=0, sim_id=0, save_path='', ckp=0, ckp_ep=0, tb_log=False, tb_layout=None,
                 explaining=(0, 0), trace_args=(5, 1, 0, 1e6), vax_args=None, cluster_args=(0, None), prev_test_feat=False, mix_learner=False, set_infected_attr=False,
                 **kwargs):
        self.control_iter = self.control_day = -1
        self.scorer_type = 0
        self.computed_measures = None
        self.inf_p = None
        self.n_nodes = n_nodes
        self.n_test = int(n_test if n_test >= 1 else n_test * n_nodes)
        self.n_trace = int(n_trace if n_trace >= 1 else n_trace * n_nodes)
        self.n_vax = int(n_vax if n_vax >= 1 else n_vax * n_nodes)
        self.mix_learner = mix_learner
        self.valid_for_trace = self.valid_for_vax = None
        self.recollection = recollection
        self.see_all_uninfected = see_all_uninfected
        self.debug_print = debug_print
        self.episode = episode
        self.eps = epsilon
        self.sim_id = sim_id
        self.ckp_ep = ckp_ep
        self.ckp = ckp
        self.tb_log = tb_log
        self.tb_layout = tb_layout
        self.loss_log = 0
        self.explaining = [i and j for i, j in zip((tester, tracer), explaining)]
        if any(self.explaining):
            self.display = __import__('IPython').display.display
            self.ax = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tester = (lambda net, nodes, size, freq=None: self.control_test(net, nodes, size)) if tester \
                        else (lambda net, nodes, size, freq=None: self.rng.choice(nodes, size, replace=False))
        self.tracer = (lambda net, nodes, size, freq=None: self.control_trace(net, nodes, size, freq)) if tracer \
                        else (lambda net, nodes, size, freq=None: self.rng.choice(nodes, size, replace=False))
        self.vaxer = (lambda net, nodes, size, freq=None: self.control_vax(net, nodes, size, freq)) if vaxer \
                        else (lambda net, nodes, size, freq=None: self.rng.choice(nodes, size, replace=False))
        
        self.trace_start, self.trace_latency, self.trace_ignore, self.trace_till = trace_args
        self.vax_start, self.vax_latency, self.vax_ignore, self.vax_till = vax_args if vax_args else trace_args
        self.trace_till = int(self.trace_till if self.trace_till >= 1 else self.trace_till * n_nodes)
        self.vax_till = int(self.vax_till if self.vax_till >= 1 else self.vax_till * n_nodes)
        self.vax_history = set()
        self.cluster_latency, *self.clustering_args = cluster_args
        if not self.cluster_latency:
            self.cluster_latency = self.vax_latency
        self.cluster_start = min(self.trace_start, self.vax_start) - self.cluster_latency
        self.keep_pos_until = keep_pos_until if keep_pos_until >= 0 else 1e9
        self.keep_neg_until = keep_neg_until if keep_neg_until >= 0 else 1e9
        # number of pos/neg added at each control_iter
        self.pos_added = deque()
        self.neg_added = deque()
        # recent negatives and positives history
        self.neg = deque()
        self.pos = deque()
        self.net_changed = True
        self.all_nodes = None
        self.set_infected_attr = set_infected_attr
        self.prev_test_feat = prev_test_feat
        # dynamic node features -> (untested, pos, neg, ever_pos)
        self.dynamic_feat = np.zeros((self.n_nodes, 4), dtype=np.float32)
        # mark all as untested initially
        self.dynamic_feat[:, 0] = 1
        self.norm_hops = 1 / (n_nodes - 1) if k_hops < 0 else 1
        self.k_hops = abs(k_hops)
        self.known_inf_neigh = np.zeros((self.n_nodes, self.k_hops), dtype=np.float32) if k_hops else None
        # the `writer` object will be populated with an instance of a logger, if appropriate arguments are provided
        self.writer = None
        # give the agent a name if not given
        if name is None:
            name = type(self).__name__
        if ckp_ep > 0 or ckp > 0:
            self.save_path = save_path + name + "/"
            os.makedirs(self.save_path, exist_ok=True)
            if tb_log:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.save_path+TB_LOCATION)
                if ckp > 0:
                    self.writer.add_custom_scalars(tb_layout)
        
    def control(self, net, control_day=0, initial_known_ids=(), net_changed=True, missed_days=0):
        """
        Performs the control agent's ranking actions for a given day in the simulation. 
        This method is assumed to be called by the instance running the diffusion simulaton at each control iteration.

        Args:
            net (Network): The network object representing the current state of the network within the diffusion simulation.
                This needs to be created from a wrapper of a NetworkX graph object that also has the following fields that it keeps up to date:
                - node_states (list): A list of approximate node states for each node in the network, denoted with a single character (e.g. 'S', 'I', 'R', 'H', 'D').
                    This does not need to be entirely accurate, and it is employed ONLY for the following purposes:
                        + Filter out nodes that are `UNACCEPTED_FOR_TEST_TRACE` (e.g. hospitalized, dead), although the agent works adequately even without this.
                        + If `net.node_infected` is not available or `self.set_infected_attr` is True, the infection status of each node is inferred from this list. 
                        Note, during evaluation, this information is ONLY utilized for establishing whether a node is infected, and ONLY upon performing a "test".
                        During training, this information is also used for creating visualization, or providing training signals for any learning-based agents.
                - node_traced (list): A list of node tracing booleans for each node in the network, marking whether they are currently traced/isolated/immunized.
                The object can also have the following OPTIONAL fields, which are used by some agents:
                - node_list (list): A list of active node IDs in the network. If not available, the agent will use `self.all_nodes`=`list(self)` instead.
                - node_infected (list): A list of node infection booleans for each node in the network. If not provided or `self.set_infected_attr` is True,
                    the agent will use `node_states` to infer the infection status of each node, as per above.
                - UNINF_STATES (set): A set of node states that are considered uninfected. If not available, the agent will use `PERCEIVED_UNINF_STATES` instead.
            control_day (int, optional): The current day of the simulation. Defaults to 0.
            initial_known_ids (tuple, optional): A tuple of node IDs that the agent is initially aware of. Defaults to ().
            net_changed (bool, optional): Whether the network has changed since the last call to control. Defaults to True.
            missed_days (int, optional): The number of days that the agent missed since the last call to control. Defaults to 0.

        Returns:
            tuple: A tuple containing three lists: the set of newly tested positives, the list of nodes that were tested, and the list of nodes that were vaccinated.

        Notes:
            This method is assumed to be called sequentially within each control iteration. If this is not the case, the agent's internal state may be updated incorrectly.
        """
        # no testing means no possibility of isolating
        if self.n_test <= 0 and self.n_vax <= 0:
            return [], [], []
                
        self.control_day = control_day
        self.control_iter += 1
        # logic executed will at the beginning of control to update internal state
        if self.control_iter == 0:
            # remember the true IDs of all nodes
            self.all_nodes = list(net)
            # correct pos_until and neg_until for the first control day
            self.keep_pos_until += control_day
            self.keep_neg_until += control_day
            # ignore missed days in the first control iteration
            missed_days = 0

        # get public state of the network
        node_list = net.node_list if hasattr(net, 'node_list') else self.all_nodes
        node_states = net.node_states
        node_traced = net.node_traced
        perceived_uninf = net.UNINF_STATES if self.see_all_uninfected and hasattr(net, 'UNINF_STATES') else PERCEIVED_UNINF_STATES
         # set the `node_infected` attribute if not already set on the network, based on the perceived uninfected states and the node states
        if self.set_infected_attr or not hasattr(net, 'node_infected'):
            self.set_infected_attr = True
            net.node_infected = [s not in perceived_uninf for s in node_states]
        node_infected = net.node_infected
        # log number of infected before any control action is taken
        if self.tb_log and self.ckp > 0 and (self.ckp_ep == 0 or self.episode % self.ckp_ep == 0) and self.control_iter % self.ckp == 0: 
            self.log(sum(node_infected))
        # create a new pointer to the network that we can safely to modify later                              
        net_control = net
            
        # real number of tests/traces/vaccinations if accounting for missed days
        to_test, to_trace, to_vax = self.n_test, self.n_trace, self.n_vax
        if missed_days:
            to_test += self.n_test * missed_days
            to_trace += self.n_trace * missed_days
            to_vax += self.n_vax * missed_days
        to_trace = min(to_trace, self.trace_till)
        to_vax = min(to_vax, self.vax_till)
        # dequeue from the recent history of positives and negatives those that were tested too long ago
        neg = self.neg
        pos = self.pos
        if to_test > 0:
            # timestamp for dequeuing pos/neg that were tested too further ago
            pos_from = control_day - self.keep_pos_until - 1
            neg_from = control_day - self.keep_neg_until - 1
            # dequeue from the recent positives and negatives history
            if pos_from >= 0:
                n_add = self.pos_added.popleft()
                for i in range(n_add):
                    pos.popleft()
            if neg_from >= 0:
                n_remove = self.neg_added.popleft()
                for i in range(n_remove):
                    neg.popleft()
        # will be used to count the total number of positives and negatives added in the current timestamp, for appending to the respective queues
        pos_count = neg_count = 0

        # TO-RETURN: set of newly tested positives
        new_pos = set()
        # initiate list of tested nodes
        tested = []
        # get the dynamic node features
        known_inf_neigh = self.known_inf_neigh   
        dynamic_feat = self.dynamic_feat
        # make the agent aware of `initial_known_ids`` at the start of control (iteration 0)
        if self.control_iter == 0:
            for nid in initial_known_ids:
                dynamic_feat_nid = dynamic_feat[nid]
                # mark person as getting tested in dynamic feat
                dynamic_feat_nid[0] = 0
                pos.append(nid)
                new_pos.add(nid)
                pos_count += 1
                # flip both current positive, and the historical positive positions
                dynamic_feat_nid[1] = dynamic_feat_nid[-1] = 1
                if self.k_hops:
                    # accumulate all neighbors of 'nodes' that are WITHIN k-hops away
                    visited = {nid}
                    # this keeps track of nodes at each depth k (exclusively)
                    k_nbrs = {nid}
                    for k in range(1, self.k_hops + 1):
                        # nodes at depth k-1 starting from nid
                        prev_k_nbrs = k_nbrs.copy()
                        for bfs_src in prev_k_nbrs:
                            k_nbrs.remove(bfs_src)
                            # neighbors at depth k starting from nid, with shortest-path passing through bfs_src
                            for k_nbr in net_control[bfs_src]:
                                if k_nbr not in visited:
                                    k_nbrs.add(k_nbr)
                                    visited.add(k_nbr)
                                    # increment by one the entry of the depth k neighbor 
                                    # (in its k-1 position, as indexes start from 0)
                                    known_inf_neigh[k_nbr][k - 1] += self.norm_hops
        # if the flag is False, disable prev test features for test ranking by making everybody seem untested
        if not self.prev_test_feat:
            dynamic_feat[:, 0] = 1
            dynamic_feat[:, 1:3] = 0

        # mark whether the network has changed since last call
        self.net_changed = net_changed
        if self.recollection:
            # filter nodes that we always know are active (i.e. NOT hospitalized/dead, and NOT currently traced)
            # NOTE: this currently means isolated people do not get further testing
            active = list(filter(lambda nid: node_states[nid] not in UNACCEPTED_FOR_TEST_TRACE 
                                 and not node_traced[nid] and nid not in initial_known_ids, node_list))
            # population that is valid for testing: i.e. active - recently-tested negatives
            # note this will be a no-op if self.keep_neg_until == 0 !
            valid_for_test = np.setdiff1d(active, neg, assume_unique=True) if self.recollection >= 1 else active
            if self.recollection >= 3:
                net_control = net.subgraph(valid_for_test)
            elif self.recollection >= 2:
                net_control = net.subgraph(active)
            self.all_nodes = list(net_control)
            # this setting allows for the centralities to be recomputed at each point when recollection is enabled
            if self.recollection >= 1 and self.recollection - int(self.recollection) != 0:
                self.net_changed = True
        else:
            valid_for_test = self.all_nodes
        
        if to_test > 0:
            # verify if sampling is actually possible in terms of population size, otherwise mark all nodes left as valid for tesing
            if to_test < len(valid_for_test):
                # test only people that have not been tested negative recently
                # since valid_for_test is a set, it is the inherent responsibility of the agent to return no duplicates
                tested = self.tester(net_control, valid_for_test, size=to_test)
                if self.explaining[0]:
                    coefs = self.explain(tested, True)
                    x_axis = list(range(coefs.shape[-1]))
                    if coefs.ndim == 1:
                        if self.ax is None:
                            fig, self.ax = plt.subplots(1, 1, figsize=(8,6), facecolor='w')
                        ax = self.ax
                        ax.clear()
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.bar(x_axis, coefs, width=.1)
                        ax.set_xticks(x_axis)
                        ax.set_xticklabels(tested)
                        ax.set_xlabel('Node', fontsize=15)
                        ax.set_ylabel('Importance', fontsize=15)
                    else:
                        if self.ax is None:
                            fig, self.ax = plt.subplots(1, len(coefs), figsize=(8,3), sharey=True, squeeze=False, facecolor='w')
                            fig.tight_layout()
                        x_avg = coefs[:, 1, :]
                        coefs = coefs[:, 0, :]
                        for i in range(len(coefs)):
                            node = tested[i]
                            ax = self.ax[0, i]
                            ax.clear()
                            xs = self.x[node].cpu().numpy()
                            xa = x_avg[i]
                            # sometimes eigenvector_centrality_numpy may return very small negative values that can be considered 0
                            if xs[1] < 0:
                                xs[1] = 0
                            if xa[1] < 0:
                                xa[1] = 0
                            ax.set_title(f'Day {control_day}. Chosen node {tested[i]}', fontsize=13)
                            ax.bar(x_axis, coefs[i], color='peachpuff', width=0.2)
                            ax.set_xticks(x_axis, self.feats_labels, fontsize=11, rotation=70)
                            ylim = coefs.max()
                            ticks = ax.get_xticks()
                            xpos = ticks[5] + .5
                            prev_test = len(xs) - 3
                            frmt = lambda x, j: ('{:04.1f}' if j == 7 or j == 8 else '{:.1f}').format(x if j < prev_test else round_decimals_up(x, 1))
                            ax.text(xpos+i*.5, -ylim/2, ' '.join((frmt(xs[j], j) for j in range(len(ticks)))), 
                                    size=10.5, ha='center', bbox=dict(facecolor='lightblue', alpha = 0.7))
                            ax.text(xpos+i*.5, -ylim/2-ylim/10, ' '.join((frmt(xa[j], j) for j in range(len(ticks)))), 
                                    size=10.5, ha='center', bbox=dict(facecolor='coral', alpha = 0.7))
                            ax.set_ylim(0, ylim)
                            self.ax[0,0].set_ylabel(r'$\beta$', fontsize=12)
                            ax.tick_params(axis='y', labelsize=11)
                            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                            ax.legend(handles=[mpatches.Patch(color='lightblue', alpha=0.7, label='Node features'), mpatches.Patch(color='coral', alpha=0.7, 
                                                                                                                                   label='Subgraph features')], 
                                                                                                                                   fontsize=11)
                    if self.explaining[0] == 1:
                        self.display(plt.figure(plt.get_fignums()[-1]))
                    elif self.explaining[0] == 2:
                        plt.savefig(f'fig/batch/explain/{control_day}.png', bbox_inches='tight')

            else:
                tested = valid_for_test
                # since all nodes will be tested, disable tracing/vax for this timestamp
                to_trace = to_vax = 0
                
            # After the model has done the test ranking, we can safely mark all as untested for the next timestamp (only the selected tests will modify this)
            dynamic_feat[:, 0] = 1
            dynamic_feat[:, 1:3] = 0
            # partition tested into positives and negatives
            for nid in tested:
                dynamic_feat_nid = dynamic_feat[nid]
                # mark person as getting tested in dynamic feat
                dynamic_feat_nid[0] = 0
                # here we actually perform the test to see whether the node is infected. Could also do this through the commented out line.
                # if node_states[nid] not in perceived_uninf:
                if node_infected[nid]:
                    # flip both current positive, and the historical positive positions
                    dynamic_feat_nid[1] = dynamic_feat_nid[-1] = 1
                    # updating all-time positives and the known_inf_neigh array happens only once for each nid
                    if not node_traced[nid]:
                        new_pos.add(nid)
                        pos.append(nid)
                        pos_count += 1
                        if self.k_hops:
                            # accumulate all neighbors of 'nodes' that are WITHIN k-hops away
                            visited = {nid}
                            # this keeps track of nodes at each depth k (exclusively)
                            k_nbrs = {nid}
                            for k in range(1, self.k_hops + 1):
                                # nodes at depth k-1 starting from nid
                                prev_k_nbrs = k_nbrs.copy()
                                for bfs_src in prev_k_nbrs:
                                    k_nbrs.remove(bfs_src)
                                    # neighbors at depth k starting from nid, with shortest-path passing through bfs_src
                                    for k_nbr in net[bfs_src]:
                                        if k_nbr not in visited:
                                            k_nbrs.add(k_nbr)
                                            visited.add(k_nbr)
                                            # increment by one the entry of the depth k neighbor 
                                            # (in its k-1 position, as indexes start from 0)
                                            known_inf_neigh[k_nbr][k - 1] += self.norm_hops
                else:
                    neg.append(nid)
                    neg_count += 1
                    dynamic_feat_nid[2] = 1
                    
            # update the count of neg added for the current timestamp
            self.pos_added.append(pos_count)
            self.neg_added.append(neg_count)
            
        else:
            # if no tests conducted, contact tracing needs to be disabled
            to_trace = 0
            
        # clustering is executed only when clustering_args are provided AND clustering_args[0] (i.e. kwarg 'display') is non-negative
        # if display = 1, always generate clusters, otherwise clusters are generated only when budgets for trace or vax are enabled
        try:
            if control_day >= self.cluster_start and (self.clustering_args[0] == 1 or self.clustering_args[0] >= 0 and \
                                                      (self.tracer and to_trace > 0 or self.vaxer and to_vax > 0)):
                _, self.inf_p = self.generate_clusters(net, *self.clustering_args)
                self.cluster_start += self.cluster_latency
        except (AttributeError, IndexError, TypeError):
            pass
        
        # TO-RETURN: set of newly traced individuals
        traced, vaxed = set(), set()
        # trace control starts after trace_start and only when to_trace > 0
        if control_day >= self.trace_start and to_trace > 0:
            # no filtering
            if self.trace_ignore == 0:
                trace_ignore = {}
            # newly-identified positives to be filtered
            elif self.trace_ignore == 1:
                trace_ignore = new_pos
            # newly-tested nodes to be filtered
            elif self.trace_ignore == 1.5:
                # tested nodes need to include the initial_known_ids in the first step
                trace_ignore = set(np.append(list(tested), initial_known_ids) if self.control_iter == 0 else tested)
            # newly-tested nodes + historic negatives to be filtered
            elif self.trace_ignore == 2:
                trace_ignore = new_pos | set(neg)
            # get neighbors of positives that are in an "active" state, and that are NOT part of the set of traced to ignore
            # this list MAY contain duplicates, which some controllers may leverage
            valid_for_trace = [neigh for nid in pos for neigh in net[nid] if not node_traced[neigh] and node_states[neigh] not in UNACCEPTED_FOR_TEST_TRACE
                               and neigh not in trace_ignore]
            # get uniques and their counts
            valid_for_trace, freq = np.unique(valid_for_trace, return_counts=True)
            # for a learning agent in a mix, missing forward passes leads to hidden state drift, so corrections steps are performed according to this boolean
            fill_missing_steps = self.mix_learner and self.valid_for_trace is not None
            # exception to the above: when the test agent is also a learning agent, the hidden state can be copied over to the tracing agent
            try:
                self.trace_agent.ranking_model.h_prev = self.test_agent.ranking_model.h_prev
                fill_missing_steps = False
            except AttributeError:
                pass
            if to_trace < len(valid_for_trace):
                self.valid_for_trace = valid_for_trace
                # some agents may exploit multiple occurances of nodes in the valid list
                traced = set(self.tracer(net_control, valid_for_trace, size=to_trace, freq=freq))
                if self.explaining[1]:
                    coefs = self.explain(traced, False)
                    ...
            else:
                if fill_missing_steps:
                    self.tracer(net_control, self.valid_for_trace, size=to_trace, freq=None)
                traced = set(valid_for_trace)
            # empirically, there seems to be a benefit of copying the trace h_prev back to the test agent
            try:
                self.test_agent.ranking_model.h_prev = self.trace_agent.ranking_model.h_prev
            except AttributeError:
                pass
            
            self.trace_start += self.trace_latency
            self.trace_till -= len(traced)
            if self.trace_till <= 0:
                print('No more contact isolation is possible!')
                self.n_trace = 0
                
        # vax control starts after vax_start and only when to_vax > 0
        if control_day >= self.vax_start and to_vax > 0:
            if self.vax_ignore == 0:
                valid_for_vax = filter(lambda nid: nid not in self.vax_history, node_list)
            elif self.vax_ignore == 1:
                valid_for_vax = filter(lambda nid: nid not in self.vax_history, active)
            elif self.vax_ignore == 1.5:
                valid_for_vax = filter(lambda nid: nid not in self.vax_history, valid_for_test)
            elif self.vax_ignore == 2:
                valid_for_vax = filter(lambda nid: nid not in self.vax_history and nid not in new_pos and nid not in traced, active)
            elif self.vax_ignore == 2.5:
                valid_for_vax = filter(lambda nid: nid not in self.vax_history and nid not in new_pos and nid not in traced, valid_for_test)
            valid_for_vax = list(valid_for_vax)
            # for learning agent in a mix, missing forward passes leads to hidden state drift, so corrections steps are performed according to this boolean
            fill_missing_steps = self.mix_learner and self.valid_for_vax is not None
            # exception to the above: when the test agent is also a learning agent, the hidden state can be copied over to the tracing agent
            try:
                self.vax_agent.ranking_model.h_prev = self.test_agent.ranking_model.h_prev
                fill_missing_steps = False
            except AttributeError:
                pass
            if to_vax < len(valid_for_vax):
                self.valid_for_vax = valid_for_vax
                vaxed = set(self.vaxer(net_control, valid_for_vax, size=to_vax))
            else:
                if fill_missing_steps:
                    self.vaxer(net_control, self.valid_for_vax, size=to_vax)
                vaxed = set(valid_for_vax)
            # empirically, there seems to be a benefit of copying the vax h_prev back to the test agent
            try:
                self.test_agent.ranking_model.h_prev = self.vax_agent.ranking_model.h_prev
            except AttributeError:
                pass

            self.vax_history |= vaxed
            self.vax_start += self.vax_latency
            self.vax_till -= len(vaxed)
            if self.vax_till <= 0:
                print('No more vaccines are available!')
                self.n_vax = 0
        
        # update entries on the network for visualization purposes
        net.control_iter, net.control_day = self.control_iter, self.control_day
        net.computed_measures = self.computed_measures
        net.tested, net.traced = tested, traced | vaxed
        return new_pos, traced, vaxed

    def log(self, total_inf):
        """
        Logs throughout the episode run the total number of infected nodes to the `writer` instance.
        If the agent has also registered a `loss_log` different than 0, this is also logged, together with the model checkpoint that achieved that.
        As such, we avoid logging losses/models for iterations where one was not calculated, while also circumventing the need to check whether the agent is learning-based.

        Args:
            total_inf (int): The total number of infected nodes in the current iteration.

        Returns:
            None
        """
        idx = f'{self.episode}:{self.sim_id}'
        self.tb_layout['Ep infected'][f'Ep {idx} infected'] = ["Multiline", [f'{idx}-inf',]]
        self.writer.add_scalar(f'{idx}-inf', total_inf, self.control_day)
        if self.loss_log != 0:
            torch = self.torch
            self.tb_layout['Ep loss'][f'Ep {idx} loss'] = ["Multiline", [f'{idx}-loss',]]
            self.writer.add_scalar(f'{idx}-loss', self.loss_log, self.control_day)
            ckp_path = self.save_path + CKP_LOCATION
            os.makedirs(ckp_path, exist_ok=True)
            torch.save({
                'model_state_dict': self.ranking_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_inf': total_inf,
                }, ckp_path + f'm{os.environ.get("SLURM_ARRAY_TASK_ID", 0)}_e{self.episode}_s{self.sim_id}_i{self.control_day}.pt')

    def finish(self, total_inf, args):
        """
        Method to run at the end of an episode, mostly for logging relevant information through the `writer` logger and save parameter checkpoints for learning agents.

        Args:
            total_inf (float): Total infection count for the current episode.
            args (argparse.Namespace): Command-line arguments.

        Returns:
            None
        """
        if self.ckp_ep > 0 and self.episode % self.ckp_ep == 0:
            print('Logging agent information to finish iteration...')
            # we write the total inf for each `sim_id` per a single separate line after EVERY `ckp_ep` episodes
            rwd_path = self.save_path + 'agent_episode.log'
            sim_id = self.sim_id
            if os.path.isfile(rwd_path):
                with open(rwd_path, 'r') as f:
                    lines = f.readlines()
                lines += ['\n'] * (sim_id + 1 - len(lines))
                lines[sim_id] = f'{lines[sim_id].strip()} {total_inf}\n'
            else:
                lines = f'{total_inf}\n'
            with open(rwd_path, 'w') as f:
                f.writelines(lines)
            
            # if a tb writer exists, write inf
            if self.writer is not None:
                self.writer.add_scalar('total_inf', total_inf, self.episode + sim_id)
            # this gets triggered only when `self` is an SL/RL agent in learning mode
            if isinstance(self, SLAgent) and self.lr > 0:
                ckp_path = self.save_path + CKP_LOCATION
                os.makedirs(ckp_path, exist_ok=True)
                self.torch.save({
                    'model_state_dict': self.ranking_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'init_kwargs': self.ranking_model.init_kwargs,
                    'total_inf': total_inf,
                }, ckp_path + f'fi{os.environ.get("SLURM_ARRAY_TASK_ID", 0)}_e{self.episode}_s{sim_id}.pt')
            
            # complex logic for writing down the parameter settings (only executes once per Simulator run)
            # this will be different than an `args` dump within the Simulator because it is tailored to the agent's needs
            hparams_path = self.save_path + 'hparams.json'
            if not os.path.isfile(hparams_path):
                # for agent logging, we exclude some hyperparameters that are not particularly relevant for the agent's behavior
                exclude = {
                    'nettype', 'shared', 'ranking_model', 'target_model', # potentially large objects
                    'k_i', 'is_learning_agent', 'is_record_agent', # set during runtime but not particularly useful here
                    'zadd_two', 'zrem_two', 'uptake_two', 'overlap_two', 'maintain_overlap_two', # triad graph params typically not used in this context
                    'animate', 'draw', 'draw_iter', 'draw_config',  # drawing related parameters
                }
                kwargs = {k: (str(v) if isinstance(v, Iterable) else v) for k, v in vars(args).items() if k not in exclude}
                # set the nettype to 'predefined' if the nettype is not a string to avoid clunky json serialization
                kwargs['nettype'] = args.nettype if isinstance(args.nettype, str) else 'predefined'
                # rewrite agent parameters in an easier to read format, also compatible with tensorboard logging
                for k, v in args.agent.items():
                    if k not in ('ranking_model', 'target_model', 'tb_layout'):
                        kwargs[f'agent_{k}'] = str(v)
                # only write model parameters if called from an SL/RL agent
                if isinstance(self, SLAgent):
                    for k, v in self.ranking_model.init_kwargs.items():
                        kwargs[f'amodel_{k}'] = str(v)
                    
                # write args to agent checkpoint and to the tensorboard logger
                with open(hparams_path, 'w') as f:
                    json.dump(kwargs, f, indent=1)
                if self.writer is not None:
                    self.writer.add_hparams(kwargs, {'hparams/total_inf': total_inf}, run_name='hparams')   
            
    def control_test(self, net, nodes, size=10, freq=None, **kwargs):
        """
        By default, if a custom control strategy is selected here (tester=True), the logic will be controlled through 'control_all'
        However, subclasses can extend this to offer special treatment for test control.
        """
        self.testing = True
        return self.control_all(net, nodes, size, freq, **kwargs)
        
    def control_trace(self, net, nodes, size=10, freq=None, **kwargs):
        """
        By default, if a custom control strategy is selected here (tracer=True), the logic will be controlled through 'control_all'
        However, subclasses can extend this to offer special treatment for tracing control.
        """
        self.testing = False
        return self.control_all(net, nodes, size, freq, **kwargs)
    
    def control_vax(self, net, nodes, size=10, freq=None, **kwargs):
        """
        By default, if a custom control strategy is selected here (vaxer=True), the logic will be controlled through 'control_test'
        However, subclasses can extend this to offer special treatment for vaccination control.
        """
        self.testing = False
        return self.control_all(net, nodes, size, freq, **kwargs)  

    def control_all(self, net, nodes, size=10, freq=None, **kwargs):
        """
        This method provides a common logic for controlling testing, tracing and vaccination, and should be overridden if no special behavior
        is expected for either of these control processes, as by default they are all falling back to this method.
        """
        raise NotImplementedError('This method was not implemented for this type of Agent. It may be the case that this Agent cannot control all intervention processes.')
        
    def explain(self, nid, test=True):
        """
        This method provides an explanation for the ranking prediction made for a given node in the graph.

        Args:
            nid (int): The ID of the node to explain.
            test (bool): Whether we explain a testing decision or a tracing/immunization decision.

        Raises:
            NotImplementedError: If the method is not implemented for the current agent.
        """
        raise NotImplementedError('This method was not implemented for this type of Agent. It may be the case that this Agent cannot provide per-node explanations.')
            

class MixAgent(Agent):
    """
    A class representing a mixed agent that can perform testing, tracing, and vaccination.

    Attributes:
        test_agent (Agent): an instance of an Agent subclass that performs testing
        trace_agent (Agent): an instance of an Agent subclass that performs tracing
        vax_agent (Agent): an instance of an Agent subclass that performs vaccination
        log_agent (int): the agent to use for logging (0: tester, 1: tracer, 2: vaxer)
        mix_learner (bool): a boolean indicating whether the agent is a mix learner
        computed_measures (dict): a dictionary containing the computed measures of the log agent, to be used for logging/displaying purposes
    """    
    def __init__(self, test_type='centrality', trace_type='centrality', vax_type='centrality', log_agent=0, cluster_agent=0, **kwargs):
        super().__init__(**kwargs)
        self.test_agent = AGENT_TYPE[test_type](**kwargs, mix_learner=False)
        mix_tracer = mix_vaxer = False
        self.trace_agent = self.vax_agent = None
        if kwargs.get('n_trace', 0) and kwargs.get('tracer', False):
            if trace_type in ('sl', 'rl'):
                self.mix_learner = True
                mix_tracer = True
            self.trace_agent = AGENT_TYPE[trace_type](**kwargs, mix_learner=mix_tracer)
        if kwargs.get('n_vax', 0) and kwargs.get('vaxer', False):
            if vax_type in ('sl', 'rl'):
                self.mix_learner = True
                mix_vaxer = True
            self.vax_agent = AGENT_TYPE[vax_type](**kwargs, mix_learner=mix_vaxer)
        # set the log agent; if one cannot be set, raise an error since the logging behavior should be readily utilizable by default
        if log_agent == 0:
            self.log_agent = self.test_agent
        elif log_agent == 1:
            if self.trace_agent is None:
                raise ValueError('Cannot use the tracer as a log agent when tracing prioritization is not enabled!')
            self.log_agent = self.trace_agent
        else:
            if self.vax_agent is None:
                raise ValueError('Cannot use the vaxer as a log agent when vaccination prioritization is not enabled!')
            self.log_agent = self.vax_agent
        # set the agent to be used for clustering; if one cannot be set, set it to None since clustering is not supported by all combinations of agents
        if cluster_agent == 0 and isinstance(self.test_agent, SLAgent):
            self.cluster_agent = self.test_agent
        elif cluster_agent == 1 and isinstance(self.trace_agent, SLAgent):
            self.cluster_agent = self.trace_agent
        elif cluster_agent == 2 and isinstance(self.vax_agent, SLAgent):
            self.cluster_agent = self.vax_agent
        else:
            self.cluster_agent = None
        
    def update_subagent(self, subagent):
        """
        Update the subagent with the current state of the parent agent.

        Args:
            subagent (Agent): The subagent to update.

        Returns:
            None
        """
        # counters and flags for current control iteration
        subagent.control_iter += 1
        subagent.control_day = self.control_day
        subagent.net_changed = self.net_changed
        # dynamic node features at this point
        subagent.dynamic_feat = self.dynamic_feat
        subagent.known_inf_neigh = self.known_inf_neigh
        subagent.inf_p = self.inf_p
        # the indices of all nodes are lazily initialized in 'control', so they need to be copied over in subagents
        subagent.all_nodes = self.all_nodes
        
    def update_agent(self):
        """
        Update the MixAgent's state with the latest information from the chosen `log_agent` after a ranking step has been performed.
        Note, `update_agent` is called after every `control_test`, `control_trace`, and `control_vax` call, but only the state of the `log_agent` is considered.
        As such, the information recorded here on the MixAgent is to be used only for visualization and logging purposes.

        Returns:
            None
        """
        self.computed_measures = self.log_agent.computed_measures

    def control_test(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Runs the control ranking action for testing of the `test_agent`.
        
        Args:
            net (Network): The network to test.
            nodes (list): The nodes to test.
            size (int): The number of nodes to return in the ranking.
            freq (list): A list of frequencies for each node in `nodes`. If None, this is usually ignored.
            **kwargs: Additional keyword arguments to pass to the test agent's `control_test` method.
        
        Returns:
            to_test (list): The ranking result for testing.
        """
        self.update_subagent(self.test_agent)
        to_test = self.test_agent.control_test(net, nodes, size, freq, **kwargs)
        self.update_agent()
        return to_test
        
    def control_trace(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Runs the control ranking action for tracing of the `trace_agent`.
        
        Args:
            net (Network): The network to trace.
            nodes (list): The nodes to trace.
            size (int): The number of nodes to return in the ranking.
            freq (list): A list of frequencies for each node in `nodes`. If None, this is usually ignored.
            **kwargs: Additional keyword arguments to pass to the trace agent's `control_trace` method.
        
        Returns:
            to_trace (list): The ranking result for tracing.
        """
        self.update_subagent(self.trace_agent)
        to_trace = self.trace_agent.control_trace(net, nodes, size, freq, **kwargs)
        self.update_agent()
        return to_trace
    
    def control_vax(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Runs the control ranking action for vaccination of the `vax_agent`.
        
        Args:
            net (Network): The network to vax.
            nodes (list): The nodes to vax.
            size (int): The number of nodes to return in the ranking.
            freq (list): A list of frequencies for each node in `nodes`. If None, this is usually ignored.
            **kwargs: Additional keyword arguments to pass to the vax agent's `control_vax` method.
        
        Returns:
            to_vax (list): The ranking result for vaccination.
        """
        self.update_subagent(self.vax_agent)
        to_vax = self.vax_agent.control_vax(net, nodes, size, freq, **kwargs)
        self.update_agent()
        return to_vax
    
    def generate_clusters(self, *args, **kwargs):
        """
        Generates clusters based on the given arguments by delegating the call to the `log_agent`.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple or None: The result of generating clusters for the log agent.
        """
        return self.cluster_agent.generate_clusters(*args, **kwargs)
    
    def finish(self, *args, **kwargs):
        """
        Method to run at the end of an episode, mostly for logging relevant information. This is delegated to the `log_agent`.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the `log_agent.finish` method.
        """
        return self.log_agent.finish(*args, **kwargs)
    
    def explain(self, nid, test=True):
        """
        Returns an explanation for the given node ID from the `log_agent`.

        Args:
            nid (int): The ID of the node to explain.
            test (bool, optional): Whether to use the test agent or the trace agent. Defaults to True.

        Returns:
            Explanation: An explanation for the given node ID.
        """
        return self.log_agent.explain(nid, test)
    

class AcquaintanceAgent(Agent):
    """
    An agent that selects nodes in a network based on acquanitance sampling (neighbor of a random selection).

    Attributes:
        rng (numpy.random.Generator): A random number generator.
    """
    def control_test(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Selects nodes in the network based on on acquanitance sampling.

        Args:
            net (Network): The network to select nodes from.
            nodes (list): The list of nodes to select from.
            size (int): The number of nodes to select.
            freq (list): A list of frequencies for each node in `nodes`. Unused argument in this agent.
            **kwargs: Additional keyword arguments.

        Returns:
            set: The set of selected nodes.
        """
        chosen = set()
        for nid in self.rng.choice(nodes, size, replace=False):
            neigh = net[nid]
            if set(neigh) - chosen:
                entry = self.rng.choice(neigh)
                while entry in chosen:
                    entry = self.rng.choice(neigh)
                chosen.add(entry)
        return chosen

    def control_vax(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Selects nodes in the network based on acquanitance sampling.

        Args:
            net (Network): The network to select nodes from.
            nodes (list): The list of nodes to select from.
            size (int): The number of nodes to select.
            freq (list): A list of frequencies for each node in `nodes`. Unused argument in this agent.
            **kwargs: Additional keyword arguments.

        Returns:
            set: The set of selected nodes.
        """
        return self.control_test(net, nodes, size=size, freq=freq, **kwargs)

    def explain(self, nid, test=True):
        """
        This random agent does not have a specific explanation method. Therefore, it returns an array of zeros.

        Args:
            nid (int or list): The node ID(s) to explain.
            test (bool): Whether we explain a testing decision or a tracing/immunization decision. Unused argument.

        Returns:
            numpy.ndarray: An array of zeros with the same length as `nid`.
        """
        return np.zeros((len(nid),))

       
class FrequencyAgent(Agent):
    """
    Agent that selects nodes based on their frequency of appearing in a given list of nodes.
    This is enabled only for tracing, where the `freq` argument of `control_trace` is a list of frequencies of appearing in the neighborhood of infected nodes.
    
    Attributes:
        computed_measures (array-like): Array of shape `(n_nodes,)` containing the frequency supplied in `freq` for each node.
        backward (int): The type of backward tracing proxy to use. Can be:
            0: sampling based on frequency (freq baseline)
            1: greedy selection of most frequency (backward tracing baseline)
            2: greedy selection of most frequency, but with a decreasing index sorting order
            3: greedy selection of most frequency, but with a random index sorting order as given by `Counter.most_common`.
    """    
    def __init__(self, backward=0, **kwargs):
        super().__init__(**kwargs)
        self.backward = backward
    
    def control_trace(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Selects a subset of nodes from the given list based on their frequency of occurrence.

        Args:
            net (Network): The network object. Unused in this type of agent.
            nodes (list): A list of node IDs to select from.
            size (int): The number of nodes to select.
            freq (list): A list of node frequencies. If None, the frequencies are computed.
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of selected node IDs.
        """
        self.computed_measures = freq
        if self.backward == 1:
            return nodes[np.argsort(freq)[-size:]]
        elif self.backward == 2:
            freq = [f * 10 - i for i, f in enumerate(freq)]
            return nodes[np.argsort(freq)[-size:]]
        elif self.backward == 3: 
            return [entry[0] for entry in Counter(nodes).most_common(size)]
        return self.rng.choice(nodes, p=freq/freq.sum(), size=size, replace=False)
    
    def explain(self, nid, test=False):
        """
        Returns the `computed_measures`=`freq` for the given node ID as explanations for its choice.

        Args:
            nid (int): The ID of the node to retrieve computed measures for.
            test (bool, optional): Whether we explain a testing decision or a tracing/immunization decision. Defaults to False.

        Returns:
            list: A list containing the computed measure (i.e. frequency) of the given node ID.
        """
        return self.computed_measures[nid]
            
    
class MeasureAgent(Agent):
    """
    An agent that ranks nodes in a network based on their `computed_measurement`, returning the top-k through using a min heap in most cases.
    Exception to the above are RL agents in training mode, which output a sample of `size` nodes from `compute_measures` that are returned directly (no top-k selection).
    This is a superclass for all centrality-based and learning-based agents.

    Attributes:
        use_freq (bool): Whether to use the frequency of nodes in the network when ranking them.
        ranker_sign (int): The sign of the ranker measurements. If 1, the higher the measurement, the higher the rank. If -1, the opposite is true.
        computed_measures (Iterable): The computed measures for each node in the network.
        scorer_type (int): The type of scorer used by the agent. If 2, the agent is an RL agent in training mode, which samples nodes instead of doing a top-k selection.
    """    
    def __init__(self, use_freq=False, max_heap=True, **kwargs):
        super().__init__(**kwargs)
        self.use_freq = use_freq
        self.ranker_sign = 1 if max_heap else -1
        
    def control_all(self, net, nodes, size=10, freq=None, **kwargs):
        """
        Returns a list of node IDs of size `size` that the agent considers the most important to control, based on the node-level `computed_measures`.

        Args:
            net (Network): The network object. Unused in this type of agent.
            nodes (list): A list of node IDs to select from.
            size (int): The number of nodes to select.
            freq (list): A list of node frequencies. If None, the frequencies are computed.
            **kwargs: Additional keyword arguments.

        Returns:
            list: The list of node IDs that the agent considers the most important to control, according to `computed_measures`.

        Notes:
            The node-level measures are stored in `computed_measures` and are updated only if `update_condition` returns True.
            Unless `scorer_type` = 2, the method uses a min heap to establish the best `size` nodes according to `computed_measures`, returning them in the end.
            If `scorer_type` = 2, the method returns the `computed_measures` directly, which contain a sample of `size` nodes.
        """
        if self.update_condition():
            self.computed_measures = self.compute_measures(net, nodes, size)
        if self.scorer_type == 2:
            return self.computed_measures
            
        rankers = self.computed_measures
        sign = self.ranker_sign
        # init priority queue to return best scored of length `size`
        pq = []
        if self.use_freq and freq is not None:
            for i, nid in enumerate(nodes):
                if len(pq) < size:
                    heapq.heappush(pq, (sign * rankers[nid] + freq[i], nid))
                else:
                    heapq.heappushpop(pq, (sign * rankers[nid] + freq[i], nid)) 
        else:
            for nid in nodes:
                if len(pq) < size:
                    heapq.heappush(pq, (sign * rankers[nid], nid))
                else:
                    heapq.heappushpop(pq, (sign * rankers[nid], nid))
        # The following block of code can be used to print information about the simulation state, the agent's decisions and the corresponding measures
        if self.debug_print:
            print('DAY ' + str(self.control_day))
            print('-----------')
            print('Overall measures:', rankers, '-----------')
            print('Negatives on day ' + str(self.control_day - 1), self.neg, '-----------')
            print('Chosen node states:')
            print([(nid, net.node_states[nid], net.node_traced[nid]) for nid in nodes])
            print('----------- PQ:', pq)
            print('----------- Return:', [entry[1] for entry in pq])
            print('==================')

        # return the nid corresponding to the most important entries in pq
        return [entry[1] for entry in pq]
    
    def update_condition(self):
        """
        Subclasses can override this if the measurement which guides the nodes ranking only needs sporadic updates

        Returns:
            bool: Whether the agent needs to recompute the measures or not. Returns True by default.
        """
        return True
    
    def explain(self, nid, test=True):
        computed = np.fromiter((self.computed_measures[n] for n in nid), float) if isinstance(self.computed_measures, dict) else self.computed_measures[nid]
        return computed
     
    def compute_measures(self, net, nodes, size=10):
        """
        Computes a measurement to compare nodes. This will be implemented differently by each centrality-based and learning-based agent.

        Args:
            net (Network): A network object. Unused in this type of agent.
            nodes (list): A list of node IDs to select from. Unused in this type of agent.
            size (int): The number of nodes that ultimately need to be selected. Unused in this type of agent.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError('A measurement to compare nodes needs to be implemented.')
        
        
class ClusterAgent(MeasureAgent):
    """
    A MeasureAgent that accumulates over `accum_over` steps the values in `self.inf_p`, returning the accumulated value as its `computed_measures`.
    This is used for creating cluster-based agents, where the `inf_p` is the infection probability of the cluster.

    Attributes:
        accum_over (int): The number of iterations to accumulate `inf_p` over.
        inf_p_accum (float): The accumulated `inf_p` values.
    """
    def __init__(self, accum_over=0, **kwargs):
        self.accum_over = accum_over
        self.inf_p_accum = None
        super().__init__(**kwargs)

    def compute_measures(self, net, nodes, size=10):
        """
        Computes the accumulated `inf_p` measures for the given network and nodes.

        Args:
            net (Network): A network object. Unused in this type of agent.
            nodes (list): A list of node IDs to select from. Unused in this type of agent.
            size (int): The number of nodes that ultimately need to be selected. Unused in this type of agent.

        Returns:
            inf_p_accum (np.ndarray): The accumulated infection probability to be used as a ranking measure.
        """
        if self.inf_p is None:
            raise ValueError('The infection probability needs to be computed before the ClusterAgent can be used.')
        if self.accum_over == 0 or self.control_iter % self.accum_over == 0:
            self.inf_p_accum = self.inf_p
        else:
            self.inf_p_accum += self.inf_p
        return self.inf_p_accum
    
        
class CentralityAgent(MeasureAgent):
    """
    A MeasureAgent that computes any networkx node measure, returning this as its `computed_measures`.
    This is used for creating centrality-based agents, where the `computed_measures` is the centrality measure of the node.

    Attributes:
        measure (str): The type of centrality measure to compute. Default is 'degree'. Supported values include:
            'degree', 'eigenvector', 'closeness', 'betweenness', 'katz', 'communicability',
            'harmonic', 'load', 'subgraph', 'percolation', 'current_flow', 'random_walk',
            'pagerank', 'second_order', 'third_order', 'fourth_order', 'fifth_order'.
        computed_measures (list or None): The list of computed centrality measures for the last network and node subset.
            None if no measures have been computed yet.
        net_changed (bool): Whether the network has changed since the last computation. If `compues_measures` is not None, a value of False means no recomputation is needed.
        n_nodes (int): The number of nodes in the network.

    Notes:
        Since we rely on the networkx API, most of the metrics will be computed for the entire graph as restricting the computation to node subsets is not supported.
    """
    def __init__(self, measure='degree', **kwargs):
        self.measure = measure
        super().__init__(**kwargs)
              
    def update_condition(self):
        """
        Determines whether the agent needs to (re)compute metrics by establishing whether the network has changed or the `computed_measures` have not yet been computed.

        Returns:
            bool: True if the network has changed or the `computed_measures` are None, False otherwise.
        """
        return self.net_changed or self.computed_measures is None
            
    def compute_measures(self, net, nodes, size=10):
        """
        Compute the networkx node measures for a given network, and then return them. These are obtained by calling the corresponding networkx API.
        The method first attempts to call a method named '{measure}_centrality', and if it fails it falls back to a method with the name '{measure}'.

        Args:
            net (Network): The network to compute the measures on.
            nodes (list): The subset of nodes for which the measures are needed. Unused in this type of agent.
            size (int): The number of nodes that ultimately need to be selected. Unused in this type of agent.

        Returns:
            list: A list of centrality measures

        Notes:
            By relying on the networkx API, we are generally limited to computing the metrics on the entire graph. 
            However, it is possible to achieve more efficient implementations for some metrics, e.g. degree, by computing them only for the subset of `nodes`.
        """
        try:
            centr_f = getattr(nx, self.measure + '_centrality')
            centr = centr_f(net)
        except AttributeError:
            centr = getattr(nx, self.measure)(net)
        # for the special case of eigenvector-based measures, where power iteration is performed, attempt the computation again with a higher max_iter
        except nx.PowerIterationFailedConvergence:
            centr = centr_f(net, max_iter=1000)
        # the supplied network may actually be a subgraph, so we need to make sure the node measures are available for the entire spectrum of node IDs.
        if len(net) != self.n_nodes:
            centr = np.fromiter((centr.get(i, 0) for i in range(self.n_nodes)), float)
        return centr
        
        
class WeightAgent(CentralityAgent):
    """
    A WeightAgent is a type of CentralityAgent that computes a weighted degree centrality based on the `weight` edge attributes.

    Attributes:
        n_nodes (int): The total number of nodes in the network.
    """
    def compute_measures(self, net, nodes, size=10):
        """
        Computes the centrality of each node in the given network based on its weighted degree.

        Args:
            net (Network): The network to compute centrality measures for.
            nodes (list): The subset of nodes for which the measures are needed. Unused in this type of agent.
            size (int): The number of nodes that ultimately need to be selected. Unused in this type of agent.

        Returns:
            centr (list or nx.DegreeView): An object that can be indexed with a node ID to obtain its corresponding centrality measure.
        """
        centr = net.degree(weight='weight')
        # the supplied network may actually be a subgraph, so we need to make sure the node measures are available for the entire spectrum of node IDs.
        if len(net) != self.n_nodes:
            centr = [centr.get(i, 0) for i in range(self.n_nodes)]
        return centr
    
    
class NeighborsAgent(MeasureAgent):
    """
    A class representing an agent that returns the known number of infected neighbors for each node in the network supplied to the interfacing method `control`.
    The latter method is also responsible for calculating the quantity returned by this agent as its `computed_measures`.
    """
    def compute_measures(self, net, nodes, size=10):
        """
        Returns the known number of infected neighbors of each node in the network initially supplied to the interfacing method `control`.
        This can span multiple hops away, according to the `k_hops` parameter.

        Args:
            net (Network): The network to compute centrality measures for. Unused in this type of agent.
            nodes (list): The subset of nodes for which the measures are needed. Unused in this type of agent.
            size (int): The number of nodes that ultimately need to be selected. Unused in this type of agent.

        Returns:
            list: An list of infected neighbors count.
        """
        return self.known_inf_neigh.tolist()


class SLAgent(MeasureAgent):
    """
    A learning-based agent that uses a `ranking_model` to predict the importance of nodes in a graph.
    By default, this is a supervised learning agent, but by providing `rl_sampler`, it becomes a reinforcement learning agent.

    Args:
        ranking_model (dict or None): A dictionary containing the configuration for the ranking model, or None to use the default configuration.
        target_model (Model or None): A target model for the agent to update during training, or None to disable target model updates.
        static_measures (tuple): A tuple of static measures to use for node feature engineering.
        weigh_up (bool): Whether to weigh up the corresponding measure for each factor supplied in static_measures.
        gpu (bool): Whether to use GPU training or not.
        lr (float): The learning rate for the optimizer. If this is 0, the agent will be assumed to be in evaluation mode.
        optimizer (str): The optimizer to use for training.
        schedulers (list): A list of learning rate schedulers to use for training.
        grad_clip (float): The maximum gradient norm to clip to during training.
        index_weight (int): The index of the feature tensor to use for SL loss weighting, or -1 to disable loss weighting.
            For degree weighted loss `index_weight` needs to be set to 0, since this is the first node feature in the current implementation.
        pos_weight (bool): If enabled, this will provide a weight for the loss term as a way to partially mitigate class imbalance in SL training.
        batch_size (int): The batch size to use for training.
        batch_overlap (int): The number of overlapping nodes to use between batches.
        epochs (int): The number of epochs to train for.
        eps (float): The epsilon value to use for the optimizer.
        edge_forget_after (int): The number of iterations to wait before forgetting edges.
        edge_replace_prev (bool): Whether to remove from `edge_index` the entries that match the current edges when building the temporal multigraph.
            This effectively ensures that we do not keep duplicate edges in the temporal multigraph.
        ignore_edge_change (bool): Whether to ignore changes in edge attributes.
        mark_delay_same_edges (bool): Whether to mark edges that have the same source and destination nodes.
        rl_sampler (str or None): The reinforcement learning sampler to use, or None to disable reinforcement learning.
        online (int): Whether to use online or offline reinforcement learning. This is an overloaded parameter, and can be:
            - 0: offline RL
            - 1: online RL, where the old state is not reevaluated with the current model.
            - (1, 2): online RL, where the old state is not reevaluated with the current model, but the old logp is calculated using the TARGET model.
            - [2,3): online RL, where the old state is reevaluated with the current model to calculate the old logp.
            - [3, 4): online RL, where the old state is reevaluated with the current model to calculate the old state value
            - >= 4: online RL, where the old state is reevaluated with the current model to calculate both the old logp and the old state value.
            By default, the online training steps are performed only on uneven steps, with even steps only providing values for the loss computation.
            However, in the special setting when online*10%10 = 5 (i.e. the tenths value is 5), the training steps are performed at every increment.
            Note, this special setting does NOT generally work for online < 2 since there would be conflicts in the gradient computation.
        rl_args (tuple): A tuple of arguments to use for reinforcement learning. In order, these arguments are:
            - gamma (float): The discount factor.
            - lamda (float): The generalized advantage estimation factor.
            - reward_scale (float): The reward scaling factor. If positive, we scale the default negative of the total number of infected.
                If negative, we scale the total number of susceptible.
            - actor_loss (str): The reinforcement learning actor loss to use. Can be 'ppo' or 'a2c'.
            - clip_ratio (float): The value to use for clipping the ratio. The sign of this is used to determine the direction we follow w.r.t. the clipped loss.
                This should generally be negative, since we want to move in the opposite direction of the ratio term (minimize negative of logp).
            - value_coeff (float): The reinforcement learning value coefficient.
            - entropy_coeff (float): The reinforcement learning entropy coefficient.
            - ce_coeff (float): A coefficient for adding up to the total loss a custom loss (e.g. CE) on the predicted node labels. By default, this is disabled.
            - eligibility (int): Marks whether eligibility traces are used. If 0, no eligibility traces are used. If 1, eligibility traces are used to
            accumulate {param}.grad. Else, eligibility traces are used to accumulate {param}.grad/{delta_value}.
            - target_update (int): The number of episodes to wait before updating the target value model.
        explain_args (tuple): A tuple of arguments to use for explanations. These represent a sequence of parameters for initializing the `GraphLIME` object.
        pred_vars (tuple): A tuple of length 2 or 3 featuring prediction type variables to use for message passing during evaluation:
            - which_edge: controls which edges to use in message passing: 0 for temporal graph, 1 for current edges, 2 both.
            - mask_pred: controls whether subgraphing on candidate nodes occurs: 0 disabled, 1 enabled for temporal graph, 2 enabled for both.
            - mask_pred_mix: can be omitted; provides means for subgraphing to be disabled for tracing or vaccination processes (remaining just for testing).
            Default is (2, 1), which means that both temporal and current edges are used for message passing, but subgraphing is enabled just for the former,
            thus allowing for positive-test related features to spread through the Diffusion module. No exception for tracing or vaccination in this case.
        debug (bool): Whether to enable debug behavior. This adds a pointer to the agent for each ranking model and performs a top-k selection 
            every other iteration. Note, enabling this should be done only when debugging the agent, being incompatible with multiprocessing.
            Also note, this is different to the `debug_print` attribute, which enables printing of information about the simulation state.
        **kwargs: Additional keyword arguments to pass to the MeasureAgent constructor. In addition to the args pased to the superclass, this can also include:
            - loss (str): The custom loss function to be used by SL agents or RL agent if ce_coeff > 0. If not provided, binary cross-entropy is used.
            - optimizer_args (dict): A dictionary of additional arguments to pass to the optimizer.
            - scheduler_args (dict): A dictionary of additional arguments to pass to the scheduler.
            - action sampling parameters like: `dist_over_sample`, `zero_outsample_logits`, `logp_from_score` etc.

    """            
    def __init__(self, ranking_model=None, target_model=None, static_measures=('degree',), weigh_up=False, gpu=False, lr=0, optimizer='Adam', schedulers=None, 
                 grad_clip=0, index_weight=-1, pos_weight=False, batch_size=0, batch_overlap=4, epochs=1, eps=.5, edge_forget_after=0, edge_replace_prev=True, 
                 ignore_edge_change=False, mark_delay_same_edges=False, rl_sampler=None, online=1, rl_args=(.99, .97, 1, 'ppo', -.2, .5, .01, 0, 1, 5), 
                 explain_args=(2, 0, 5, .1, 1, 500), pred_vars=(2, 1), debug=False, **kwargs):
        # always enable tensorboard logging for SL agents since `torch` needs to be imported anyway
        kwargs['tb_log'] = True
        super().__init__(**kwargs)
        self.static_measures = static_measures
        self.weigh_up = weigh_up
        self.edge_forget_after = edge_forget_after
        self.edge_replace_prev = edge_replace_prev
        self.mark_delay_same_edges = mark_delay_same_edges
        self.ignore_edge_change = ignore_edge_change
        self.grad_clip = grad_clip
        self.online = online
        self.batch_size = batch_size
        self.batch_overlap = batch_overlap
        self.epochs = epochs
        # prediction type selection variables (used to filter edges for message passing during evaluation)
        self.pred_vars = pred_vars

        # local import of torch
        torch = self.torch = __import__('torch')
        self.gpu = gpu and torch.cuda.is_available()
        self.lr = lr
        self.rl_sampler = rl_sampler
            
        self.cluster_prev = None
        self.feats_labels = None
        # recording random features for breaking symmetry
        self.rand_emb = None
        # memory: static measures
        self.centr = None
        # memory: cummulative edges and edge attrs
        self.edge_index = None
        self.edge_attr = None
        self.edge_current = (None, None)
        
        # only importing iff debug prints is enabled
        if self.debug_print:
            self.metrics = __import__('torchmetrics').functional
            self.pd = __import__('pandas')
        
        # ranking_model is None or a dict only in evaluation mode when each Agent creates and owns a unique Model instance
        # this allows for custom behavior, such as truly separating testing and tracing Agents
        if not ranking_model:
            ranking_model = Agent.model_from_dict(k_hops=self.k_hops, static_measures=static_measures, torch_seed=self.seed)
        elif isinstance(ranking_model, dict):
            if self.mix_learner and 'initial_weights_trace' in ranking_model:
                ranking_model_copy = ranking_model.copy()
                ranking_model_copy['initial_weights'] = ranking_model['initial_weights_trace']
                ranking_model = ranking_model_copy
            ranking_model = Agent.model_from_dict(k_hops=self.k_hops, static_measures=static_measures, **ranking_model)
        else:
            # make sure previous iteration's h_prev are never utilized
            ranking_model.h_prev = None
            if target_model is not None:
                target_model.h_prev = None
        
        # if no target_model provided and this is an RL online scenario with updates for the target_model enabled, just copy the ranking_model
        # rl_args[-1] signals the update rate for the target_model (if 0, disabled)
        if lr > 0 and rl_sampler and online and target_model is None and rl_args[-1]:
            target_model = copy.deepcopy(ranking_model)
        is_target_exist = target_model is not None

        # only for debugging purposes: create a pointer to the agent in each ranking model, and make top-k run with no training every other iteration
        if debug:
            ranking_model.agent = self
            if is_target_exist:
                target_model.agent = self
            if self.episode % 2:
                self.lr = 0
                self.episode = 0
        
        device_type = ranking_model.device.type
        if self.gpu:
            if device_type == 'cpu':
                ranking_model.cuda()
                if self.rl_sampler and is_target_exist:
                    target_model.cuda()
        elif device_type != 'cpu':
            ranking_model.cpu()
            if self.rl_sampler and is_target_exist:
                target_model.cpu()
        
        # this will accumulate the loss over multiple timestamps
        self.loss_accum = torch.tensor(0)
        # remember state and action for previous timestamp (only for online)
        self.old_state_action = None
        # model explainer
        self.explainer = None
        if self.lr:
            ranking_model.train()
            self.loss = kwargs.get('loss', torch.nn.BCEWithLogitsLoss())
            self.optimizer = getattr(torch.optim, optimizer)(ranking_model.parameters(), lr=lr, **kwargs.get('optimizer_kwargs', {})) \
                                if isinstance(optimizer, str) else optimizer
            self.schedulers = []
            if schedulers:
                schedulers_args = kwargs.get('schedulers_args', [(100, .1)] * len(schedulers))
                for i, scheduler in enumerate(schedulers):
                    self.schedulers.append(getattr(torch.optim.lr_scheduler, scheduler)(self.optimizer, *schedulers_args[i]) \
                                                if isinstance(scheduler, str) else scheduler)
            # special initialization logic for RL agents
            if self.rl_sampler:
                ## special parameters for sampling actions that modify the ActionSampler's behavior and logp calculation
                # whether the distribution is created over all nodes (0), just over the candidate sample (1), or over all but the candidates masked (2)
                self.dist_over_sample = kwargs.get('dist_over_sample', 1)
                # if this is True, candidate nodes will be saved for each iteration, and then `sampler.get_logp_and_entropy` will replace non-candidate probs with approx. 0
                self.zero_outsample_logits = kwargs.get('zero_outsample_logits', False)
                # can choose to sample without replacement from the pi.Categorical dist, or sample with replacement from torch.multinomial + torch.nn.functional
                self.sample_from_dist = kwargs.get('sample_from_dist', False)
                # if sample_from_dist False: selects whether the logp values for the RL criterion are the raw logits (False) or computed from selected pi.Categorical dist (True)
                self.logp_from_score = kwargs.get('logp_from_score', True)
                # if sample_from_dist True: forces the sampling without replacement to only sample from candidate nodes (this can lead to slow or even infinite loops!)
                self.force_cndt_sample = kwargs.get('force_cndt_sample', False)
                # by default, logp will be summed across the `size` dim in the offline setting, and left unsummed in the online setting
                self.sum_logp = kwargs.get('sum_logp', not self.online)
                # create and populate an ActionSampler instance with the agent state
                from .action_sampler import ActionSampler
                self.action_sampler = ActionSampler(self)
                self.total_reward = 0
                self.scorer_type = 2
                self.gamma, self.lamda, self.reward_scale, self.actor_loss, self.clip_ratio, self.value_coeff,\
                    self.entropy_coeff, self.ce_coeff, self.eligibility, self.target_update = rl_args
                if online:
                    # eligibility trace of the loss gradients
                    self.trace = [torch.zeros_like(p.data, requires_grad=False) for p in ranking_model.parameters()]
                    # logic to delay the update of the target value network for a number of episodes
                    if is_target_exist and self.episode and self.episode % self.target_update == 0:
                        print('Changing target model...')
                        target_model.load_state_dict(ranking_model.state_dict())
                else:
                    # local import of PYG needed to construct PYG batches
                    self.pyg_data = __import__('torch_geometric').data
                    # when not online, a replay buffer will be needed
                    self.replay_buffer = ReplayBuffer()
                    # we do not know at this point the total number of batches
                    self.num_batches = 0
            else:
                # accumulating gradients over a certain period
                self.index_weight = index_weight
                self.pos_weight = pos_weight
        else:
            # if no lr has been given, then assume the model should be put into eval mode
            if ranking_model.training:
                ranking_model.eval()
            if any(self.explaining):
                from .model_utils import GraphLIME
                self.explainer = GraphLIME(ranking_model, *explain_args)
                
        # finally, assign the models as instance variables to the Agent
        self.ranking_model = ranking_model
        self.target_model = target_model

    def generate_clusters(self, net, display=0, n_clusters=7, model_inf_p=False, save_res=False, show_prev_inf=False, **kwargs):
        """
        Generates clusters of embeddings seen in the previous call, and calculates each cluster's total count and infection prob via either current time stats or an SL agent.
        We perform the clustering for the embeddings of the previous call, to make the current time stats a proxy for computing the true forward-time infection probability.
        The method can also display a dendrogram of these clusters, together with a TSNE plot of the embeddings, and save the plot if the corresponding flags are set.

        Args:
            net (Network): The network object.
            display (int): Whether to display the plot or not.
            n_clusters (int): Number of clusters to generate.
            model_inf_p (bool): Whether to use an infection probability model like SLAgent or not. If False, a true forward-time infection probability is calculated.
            save_res (bool): Whether to save the plot or not.
            show_prev_inf (bool): Whether to show previously infected nodes or not.
            **kwargs: Additional arguments.

        Returns:
            tuple: Tuple of None's if the method is called for the first time, otherwise returns the calculated infection probability and total count of each cluster.
        """
        # this first block executes only the first time the `generate_clusters`` method is called
        if self.cluster_prev is None:
            self.cluster_prev = [self.ranking_model.h_prev.cpu().numpy(), np.array(net.node_infected), set(self.neg)]
            if model_inf_p:
                self.inf_model = copy.deepcopy(self.ranking_model)
                self.inf_model.load_state_dict(self.torch.load('saved/ckp/perform/sl.pt'))
                self.inf_model.eval()
            return None, None
        else:
            from sklearn.manifold import TSNE
            from sklearn.cluster import AgglomerativeClustering
            emb, prev_inf, prev_neg = self.cluster_prev
            now_inf = np.array(net.node_infected)
            new_inf = set(np.nonzero(now_inf != prev_inf)[0])
            if model_inf_p:
                torch = self.torch
                with torch.no_grad():
                    inf_relative_prob = torch.sigmoid(self.inf_model(self.x, self.edge_index, edge_attr=self.edge_attr)[0]).numpy()
                inf_pred = np.where(inf_relative_prob > 0.5, 1, 0)
            # cluster the previous embeddings
            model = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True).fit(emb)
            # calculate infection probability and total count of each cluster
            count, inf_count = [0] * n_clusters, [0] * n_clusters
            for i, label in enumerate(model.labels_):
                count[label] += 1
                if model_inf_p:
                    inf_count[label] += inf_pred[i]
                elif i in new_inf:
                    inf_count[label] += 1
            inf_p = []
            for i, label in enumerate(model.labels_):
                inf_p.append(round(inf_count[label]/count[label], 2))
            # create the counts of samples under each dendrogram node
            counts = np.zeros(model.children_.shape[0], dtype=int)
            inf_counts = np.zeros(model.children_.shape[0], dtype=int)
            labels = [0] * model.children_.shape[0]
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = current_inf_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                        if child_idx in new_inf:
                            current_inf_count += 1
                    else:
                        current_count += counts[child_idx - n_samples]
                        current_inf_count += inf_counts[child_idx - n_samples]
                counts[i] = current_count
                inf_counts[i] = current_inf_count
                ratio = round(current_inf_count/current_count, 2)
                labels[i] = f'|{current_count} 0|' if not ratio else f'|{current_count} .{str(ratio)[2:]}|'
            
            if display:  
                fig, ax = plt.subplots(1,2, figsize=(11, 4))
                fig.tight_layout(h_pad=0,w_pad=-2.5)
                # The following lines allow for a title to be added to the dendogram, mentioning the simulation day for which current embeddings were derived.
                # fig.suptitle(f'Embeddings from day {self.control_day - self.cluster_latency}', fontsize=14)
                # fig.subplots_adjust(top=0.88)
                
                ax[0].set_title(f'New infected for days {self.control_day - self.cluster_latency}-{self.control_day}: {len(new_inf)}', fontsize=13)
                ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    x_embedded = TSNE(n_components=2, perplexity=30, early_exaggeration=20, learning_rate='auto', init='pca').fit_transform(emb)
                
                tr, ni, pi, pn, rst = [], [], [], [], []
                for nid in net.nodes:
                    if net.node_traced[nid]:
                        tr.append(nid)
                    elif nid in new_inf:
                        ni.append(nid)
                    elif show_prev_inf and net.node_infected[nid]:
                        pi.append(nid)
                    elif nid in prev_neg:
                        pn.append(nid)
                    else:
                        rst.append(nid)
                ax[0].scatter(x_embedded[rst, 0], x_embedded[rst, 1], c='lightgray', marker='o', s=30)
                ax[0].scatter(x_embedded[pn, 0], x_embedded[pn, 1], c='green', marker='x', s=30)
                ax[0].scatter(x_embedded[tr, 0], x_embedded[tr, 1], c='b', marker='+', s=30)
                ax[0].scatter(x_embedded[pi, 0], x_embedded[pi, 1], c='darkred', marker='o', s=30)
                ax[0].scatter(x_embedded[ni, 0], x_embedded[ni, 1], c='r', marker='o', s=30)
                ax[1].set_title(f'Clustering with number of clusters: {n_clusters}', fontsize=14)
                ax[1].tick_params(left=False, labelleft=False, size=8)
                plot_dendrogram(model, counts, labels, truncate_mode='lastp', p=n_clusters, ax=ax[1])
                if save_res:
                    plt.savefig(f'fig/batch/notrain/{self.control_day}.eps', bbox_inches='tight')
                    plt.savefig(f'fig/batch/notrain/{self.control_day}.png', bbox_inches='tight')
                else:
                    plt.show()
                    sleep(.8)
            # prepare for next iteration
            self.cluster_prev = [self.ranking_model.h_prev.cpu().numpy(), now_inf, set(self.neg)]
            return model.labels_, np.array(inf_p)
           
    def explain(self, nid, test=True):
        """
        Returns an explanation for the given node ID according to local feature importances fitted using GraphLIME.

        Args:
            nid (int): The ID of the node to explain.
            test (bool, optional): Whether to use the test agent or the trace agent. Defaults to True.

        Returns:
            Explanation: An explanation for the given node ID.
        """
        if self.explainer:
            return np.array([self.explainer.explain_node(n.item(), self.x, self.edge_index, edge_attr=self.edge_attr, logits=self.logits) for n in nid])
        else:
            return np.zeros((len(nid),))
        
    def finish(self, total_inf, args):
        """
        Method to run at the end of an episode, mostly for logging relevant information through the `writer` logger and save parameter checkpoints for learning agents.
        Additionally, here we also perform cleanup of objects that are no longer needed, and offline update the parameters of the `ranking_model` if this is offline RL.

        Args:
            total_inf (float): Total infection count for the current episode.
            args (argparse.Namespace): Command-line arguments.

        Returns:
            None
        """
        # cleanup objects that are no longer needed
        for obj in (self.edge_index, self.edge_attr, self.edge_current[0], self.edge_current[1], self.loss_accum):
            if obj is not None:
                obj.detach_()
                del obj
        # offline update of parameters is possible only when both lr and rl_sampler are set (i.e. offline RL agent in training mode)
        if self.lr and self.rl_sampler:
            if self.online:
                del self.old_state_action
                for obj in self.trace:
                    try:
                        obj.detach_()
                    except RuntimeError:
                        pass
                    del obj
                del self.trace
            else:
                print('Update parameters at the end of episode...')
                self.update_parameters(last_reward=-total_inf)
        if self.gpu:
            self.torch.cuda.empty_cache()
        # logic to deal with logging and saving checkpoints is already handled by the Agent superclass
        super().finish(total_inf, args)
                
    def update_parameters(self, last_reward=0):
        """
        Update the parameters of the agent's `ranking_model` using a replay buffer. This is only used for offline RL agents.

        Args:
            last_reward (float): The last reward received by the agent.

        Raises:
            ValueError: If the value of `self.epochs` is 0.
        """
        torch = self.torch
        pyg_data = self.pyg_data
        self.replay_buffer.shift_and_discount_rewards(last_reward, self.gamma, self.lamda, abs(self.reward_scale))
        device = self.ranking_model.device
        # calculate the total number of batches if unknown
        if not self.num_batches:
            self.num_batches = self.replay_buffer.n_samples // (self.batch_size - self.batch_overlap)
        
        # epochs can be either inner or outer of the batch loop
        if self.epochs < 0:
            outer_loop = lambda: tqdm_redirect(range(-self.epochs), total=-self.epochs)
            batch_loop = lambda: self.replay_buffer.sample(batch_size=self.batch_size, overlap=self.batch_overlap)
            inner_loop = 1
        elif self.epochs > 0:
            outer_loop = lambda: range(1)
            batch_loop = lambda: tqdm_redirect(self.replay_buffer.sample(batch_size=self.batch_size, overlap=self.batch_overlap), total=self.num_batches)
            inner_loop = self.epochs
        else:
            raise ValueError('The value of `self.epochs` cannot be 0')
        
        for _ in outer_loop():
            edge_index = [None] * self.batch_size
            edge_attr = [None] * self.batch_size
            self.ranking_model.h_prev = None
            for states, actions, logp_old, values, rewards, cndt_nodes in batch_loop():
                adv = torch.tensor(values, device=device)
                returns = torch.tensor(rewards, device=device)
                logp_old = torch.tensor(logp_old, device=device)
                actions = torch.stack(actions)
                if self.dist_over_sample:
                    max_len_cndt = len(max(cndt_nodes, key=len))
                    # pad nodes lists with the last value in each sequence
                    cndt_nodes = torch.stack([torch.nn.functional.pad(seq, pad=(0, max_len_cndt-len(seq)), mode='constant', 
                                                                      value=seq[-1]) for seq in cndt_nodes])
                else:
                    # free memory for unneeded list of cndt_nodes
                    cndt_nodes = None

                # establish the true batch_size (taking into consideration the last batch which may be incomplete)
                batch_size = len(states)
                # if a mismatch exists (i.e. when an incomplete batch is encountered), h_prev needs to be trimmed
                if self.batch_size != batch_size:
                    index_keep = len(self.all_nodes) * (self.batch_size - batch_size)
                    self.ranking_model.h_prev = self.ranking_model.h_prev[index_keep:]

                x, edge_index_current, edge_attr_current, edge_accumulate, y = zip(*states)
                if edge_index[-1] is None:
                    edge_index[-1] = edge_index_current[0]
                    edge_attr[-1] = edge_attr_current[0]
                data_list = []
                for i, accum in enumerate(edge_accumulate):
                    if accum:
                        edge_attr_last = edge_attr[i-1]
                        remember_indices = edge_attr_last[:, 1] < self.edge_forget_after
                        edge_index[i] = torch.cat((edge_index[i-1][:, remember_indices], edge_index_current[i]), dim=1)
                        # increase by 1 the time delay of all other timestamps (Note, the current one is yet to be appended)
                        edge_attr_last = edge_attr_last[remember_indices].clone()
                        edge_attr_last[:, 1] += 1
                        edge_attr[i] = torch.cat((edge_attr_last, edge_attr_current[i]), dim=0)
                    else:
                        edge_index[i] = edge_index[i-1]
                        edge_attr[i] = edge_attr[i-1]
                    data_list.append(pyg_data.Data(x=x[i], edge_index=(edge_index[i], edge_index_current[i]), 
                                                   edge_attr=edge_attr[i], edge_attr_current=edge_attr_current[i]))

                ## PYG batching: we pass a tuple of both edge_index and edge_index_current to ensure both get batched as adjacency matrices
                batch = self.pyg_data.Batch.from_data_list(data_list)
                if self.gpu:
                    batch = batch.to(device)
                    actions = actions.to(device)

                for _ in range(inner_loop):
                    loss = torch.tensor(0)
                    # the raw scores in y_pred are actually logits of the policy
                    y_pred, v_score = self.ranking_model(batch.x, batch.edge_index[0], batch.edge_attr, batch_idx=batch.batch, \
                                                         edge_current=(batch.edge_index[1], batch.edge_attr_current), scorer_type=2)
                    y_pred = y_pred.reshape(batch_size, -1)
                    
                    logp, entropy = self.action_sampler.get_logp_and_entropy(y_pred, actions, nodes=cndt_nodes, sum_logp=True)

                    if self.clip_ratio:
                        # calculate the actor loss
                        # allow for the loss to point towards or against logp via the sign of clip_ratio
                        # the default behavior should be: minimize a loss that points against logp, equivalent to maximizing logp
                        direction = np.sign(self.clip_ratio)
                        clip_ratio = abs(self.clip_ratio)
                        if self.actor_loss == 'ppo':
                            ratio = torch.exp(torch.clamp(logp - logp_old, max=50))
                            clip = torch.clamp(ratio, min=1-clip_ratio, max=1+clip_ratio)
                            raw_objective = torch.min(ratio * adv, clip * adv)
                        else:
                            raw_objective = logp * adv
                        loss_actor = direction * raw_objective.mean()
                        loss = loss + loss_actor
                    if self.value_coeff:
                        # calculate the critic loss
                        loss_critic = ((returns - v_score) ** 2).mean()
                        loss = loss + self.value_coeff * loss_critic
                    if self.entropy_coeff:
                        # calculate the -Entropy of the new policy
                        # model should maximize the Entropy which means minimizing -Entropy
                        loss = loss - self.entropy_coeff * entropy.mean()
                    if self.ce_coeff:
                        # multitask learning with PPO + custom loss (e.g. Cross-Entropy)
                        loss_ce = self.loss(y_pred, torch.stack(y).to(device))
                        loss = loss + self.ce_coeff * loss_ce
                    # backproing through pi.log_prob is non-deterministic on CUDA due to the use of 'scatter_add_cuda'
                    if self.gpu and self.ranking_model.deterministic:
                        torch.use_deterministic_algorithms(False)
                        loss.backward()
                        torch.use_deterministic_algorithms(True)
                    else:
                        loss.backward()

                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.ranking_model.parameters(), self.grad_clip)
                    elif self.grad_clip < 0:
                        torch.nn.utils.clip_grad_norm_(self.ranking_model.parameters(), abs(self.grad_clip))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if not self.ranking_model.detach_h:
                        # `h_prev` needs to be detached here to avoid in-place modification errors
                        # note that, for ensembles, this redirects to the children models
                        self.ranking_model.h_prev = self.ranking_model.h_prev.detach()
                    if self.schedulers:
                        for scheduler in self.schedulers:
                            try:
                                scheduler.step()
                            except TypeError:
                                scheduler.step(loss_critic)
        
    def compute_measures(self, net, nodes=None, size=10):
        """
        Computes the ranking measures of each node in the given network based on the assessment of the learning-based agent.
        This is also responsible for accumulating the temporal multigraph over multiple timestamps, and for performing training steps if the agent is in training mode.
        If `lr` is 0, this is assumed to be in evaluation mode, and the measures are returned directly from the `ranking_model`'s predictions (same for both SL/RL agents).
        Otherwise, the agent is assumed to be in training mode, with a learning rate equal to `lr`. And the logic further splits in this case:
            - For standard SL agents, this also performs the online learning step using the loss provided in kwargs['loss'] and the node infection labels.
            - For RL agents (i.e. if an `rl_sampler` was provided), this method no longer returns node measures, but directly a sample of nodes, which MeasureAgent
              knows to return directly instead of performing the top-k selection step. Depending on the `online` parameter, this method also does one of the following:
                - if online = 0, it accumulates in the offline replay buffer the states, actions, rewards, and logp_old values recorded for each timestamp.
                - if online > 0, it performs an online learning step using the RL loss and the reward received in this timestamp (check class comment for specific `online` settings).
              The RL rewards in the epidemic environment are either the negative of total number of infected or the total number of susceptible in the current state, 
              depending on the `reward_scale`, and correspond to the last action taken (which has effectively lead to the current state).

        Args:
            net (Network): The network to compute node ranking measures for, and with which the GNN training is performed.
            nodes (list): The subset of nodes for which the measures are needed.
            size (int): The number of nodes that ultimately need to be selected.

        Returns:
            np.ndarray: The predicted node measures OR a sample of nodes if this is an RL agent in training mode (i.e. scorer_type=2).
        """
        if nodes is None or len(nodes) == 0:
            nodes = list(net)
        torch = self.torch
        device = self.ranking_model.device
        if self.gpu and self.control_iter % 1 == 0:
            torch.cuda.empty_cache()
        
        x = torch.from_numpy(self.get_features(net, self.net_changed)).to(device)
        edge_accumulate = False
        edge_index, edge_attr = self.edge_index, self.edge_attr
        # if the edges have not changed since last time (or ignore_edge_change=True), at least one timepoint happened, and mark_delay_same_edges is disabled, self contains all the info needed
        if edge_index is not None and (not self.net_changed or self.ignore_edge_change) and not self.mark_delay_same_edges:
            edge_index_current, edge_attr_current = self.edge_current
        # otherwise, edge_current tensors will need to be created from the supplied (updated) network
        else:
            edges_current_time = torch.tensor(list(net.to_directed().edges.data('weight', default=1.)), device=device)
            # transform list of edges into COO format for torch geometric
            edge_index_current = edges_current_time[:, :2].long().T.contiguous()
            # mark the time delay feature as 0 for the current timestamp
            edge_attr_current = torch.nn.functional.pad(input=edges_current_time[:, -1].float().reshape(-1, 1),
                                      pad=(0, 1, 0, 0), mode='constant', value=0)
            # if no edge_index have been remembered yet, this must be the first edge entry, so update edge_index with edge_index_current
            if edge_index is None:
                edge_index = edge_index_current
                edge_attr = edge_attr_current
            # otherwise, we know that edges have changed (or they should be marked as changed), so we accumulate the temporal multigraph
            else:
                edge_accumulate = True
                # 0 means all past will be forgotten
                remember_indices = edge_attr[:, 1] < self.edge_forget_after
                # allows for edges to be replaced if they are the same as the current ones
                # this is effectively done through first removing the indices where these occur, such that we add the full current edge list
                if self.edge_replace_prev and self.edge_forget_after > 0:
                    present = []
                    current = edge_index_current.cpu().numpy()
                    row = current[0]
                    col = current[1]
                    for u, v in edge_index.T.tolist():
                        try:
                            present.append(max(u == row[v == col]))
                        except (TypeError, ValueError):
                            present.append(False)
                    remember_indices &= ~torch.tensor(present, device=device)
                edge_index = torch.cat((edge_index[:, remember_indices], edge_index_current), dim=1)
                 # increase by 1 the time delay of all other timestamps (Note, the current one is yet to be appended)
                edge_attr = edge_attr[remember_indices].clone()
                edge_attr[:, 1] += 1
                edge_attr = torch.cat((edge_attr, edge_attr_current), dim=0)
        # update the memorized temporal edge_index and edge_attr
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_current = (edge_index_current, edge_attr_current)
        
        if self.lr:
            loss = self.loss_accum
            context = nullcontext
            if self.online:
                h_prev_bu = self.ranking_model.h_prev.clone() if self.control_iter > 0 else None
            # when offline learning with an RLsampler, do not track gradients on the forward pass
            elif self.rl_sampler:
                context = torch.no_grad
            with context():
                # get the predicted infection status of all nodes (and optionally the state score, for scorer_type=2)
                y_pred, v_score = self.ranking_model(x, edge_index, edge_attr, edge_current=self.edge_current, scorer_type=self.scorer_type)

            # get the true infection status of all nodes (i.e. y_true): this will be used by agents in SL mode, but also to compute confusion matrices if needed.
            y = torch.tensor(net.node_infected, dtype=float, device=y_pred.device)
            # sometimes `node_infected` may be larger than the actual number of nodes to allow for nids to be discontinuous
            # this means that one needs to select the active nodes from the `node_infected` list to get the true list of infections
            if len(x) != len(y):
                y = y[self.all_nodes]
            # get true total number of infected: this will be used by agents in RL mode, or could be used to enhance the capabilities of SL agents if needed.
            total_inf = y.count_nonzero().item()
            
            if self.rl_sampler:
                # reward signal corresponding to the 'action' taken in the previous state, arriving in the new state
                # with the sign of reward_scale one can control whether the reward is -total_inf or total_noninf
                reward = -total_inf if self.reward_scale > 0 else self.n_nodes - total_inf
                self.total_reward += reward
                
                if self.control_iter % 2 == 0:
                    # online in (1, 2) will make y_pred be evaluated by the target_model, i.e. acting old policy recorded from even timestamps
                    if self.target_model and self.online > 1 and self.online < 2:
                        with torch.no_grad():
                            y_pred, _ = self.target_model(x, edge_index, edge_attr, edge_current=self.edge_current, scorer_type=0, h_prev=h_prev_bu)
                    # for online=1, no evaluation of previous state, and implicitly h_prev_bu, occurs at t+1; 
                    # for online=2, h_prev_bu cannot be reused at t+1 for evaluating previous state since there would be no policy change between iterations
                    elif self.online <= 2:
                        h_prev_bu = None

                nodes = torch.tensor(nodes, device=device, dtype=torch.int64)
                # Equivalent to agent.get_action(); env.step() will be delayed from this point, occurring after the Agent returns its rankings
                chosen_nodes, logp, entropy = self.action_sampler.get_action(y_pred, nodes, size, sum_logp=self.sum_logp)
                chosen_nodes = chosen_nodes.squeeze(0)
                
                if self.online:
                    if self.old_state_action and (self.control_iter % 2 == 1 or self.online * 10 % 10 == 5):
                        # evaluate target state with current model OR target model
                        if self.target_model:
                            with torch.no_grad():
                                _, v_target = self.target_model(x, edge_index, edge_attr, edge_current=self.edge_current, scorer_type=1, h_prev=None)
                        else:
                            v_target = v_score.detach()
                            
                        # old_state remembers the action taken to get to this stage, as well as the previous node and state scores
                        x, edge_index, edge_attr, edge_index_current, edge_attr_current, h_prev_bu, logp_old, v_score_old, chosen_nodes_old, nodes_old = self.old_state_action
                        # clear old_state from memory
                        self.old_state_action = None
                        # in online == 2 mode, re-evaluate with current policy the old state (rather than utilizing the new state's evaluation)
                        if self.online >= 2:
                            y_pred_old_newp, v_score_old_newp = self.ranking_model(x, edge_index, edge_attr, edge_current=(edge_index_current, edge_attr_current), scorer_type=2, h_prev=h_prev_bu)
                            if self.online >= 4:
                                y_pred = y_pred_old_newp
                                v_score_old = v_score_old_newp
                            elif self.online >= 3:
                                v_score_old = v_score_old_newp
                            else:
                                y_pred = y_pred_old_newp
                                
                        # for PPO, we want the base logp of the action remembered in old_state rather than the chosen action in the present state
                        logp, entropy = self.action_sampler.get_logp_and_entropy(y_pred, chosen_nodes_old, nodes=nodes_old, sum_logp=self.sum_logp)

                        # calculate td error graph and value
                        delta = abs(self.reward_scale) * reward + self.gamma * v_target - v_score_old
                        delta_value = delta.item()
                        if self.clip_ratio:
                            # calculate the actor loss
                            adv = delta_value
                            ratio = torch.exp(torch.clamp(logp - logp_old.detach(), min=-50, max=50))
                            # allow for the loss to point towards or against logp via the sign of clip_ratio
                            # the default behavior should be: minimize a loss that points against logp, equivalent to maximizing logp
                            direction = np.sign(self.clip_ratio)
                            clip_ratio = torch.tensor(abs(self.clip_ratio))
                            if self.actor_loss == 'ppo':
                                # equivalent mathematical formulation for actor loss for when we compute for a single action
                                raw_objective = adv * (torch.min(ratio, 1 + clip_ratio) \
                                                       if adv >= 0 else torch.max(ratio, 1 - clip_ratio))
                            elif self.actor_loss == 'a2c':
                                raw_objective = adv * logp_old
                            else:
                                raw_objective = adv * logp
                            loss_actor = direction * raw_objective.mean() # this should be sum(), but empirically works better
                            loss = loss + loss_actor
                        if self.value_coeff:
                            # calculate the critic loss
                            loss_critic = delta ** 2 if self.value_coeff > 0 else v_score_old
                            # print('Val:', self.value_coeff * loss_critic)
                            loss = loss + abs(self.value_coeff) * loss_critic
                        if self.entropy_coeff:
                            # calculate the -Entropy of the new policy
                            # model should maximize the Entropy which means minimizing -Entropy
                            loss = loss - self.entropy_coeff * entropy
                        if self.ce_coeff:
                            # multitask learning with PPO + custom loss (e.g. Cross-Entropy)
                            loss_ce = self.loss(y_pred, y)
                            loss = loss + self.ce_coeff * loss_ce

                        # accumulating gradients over a certain period IF self.batch_size is not 0
                        # for backward compatibility, we allow for updates to happen every self.control_iter (without even entries) when batch_size < 0
                        update = not self.batch_size or (self.control_iter != 1 and (self.control_iter - int(self.batch_size > 0)) % abs(self.batch_size) == 0)

                        # we retain the graph if this is NOT an optimizer.step iteration OR detach_h is disabled
                        retain_graph = not update and not self.ranking_model.detach_h
                        # backproing through pi.log_prob is non-deterministic on CUDA due to the use of 'scatter_add_cuda'
                        if self.gpu and self.ranking_model.deterministic:
                            torch.use_deterministic_algorithms(False)
                            loss.backward(retain_graph=retain_graph)
                            torch.use_deterministic_algorithms(True)
                        else:
                            loss.backward(retain_graph=retain_graph)
                        if self.eligibility:
                            for idx, p in enumerate(self.ranking_model.parameters()):
                                if p.grad is not None:
                                    grad = p.grad if self.eligibility == 1 else p.grad / delta_value
                                    self.trace[idx] = self.gamma * self.lamda * self.trace[idx] + grad
                                    p.grad = delta_value * self.trace[idx]
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_value_(self.ranking_model.parameters(), self.grad_clip)
                        elif self.grad_clip < 0:
                            torch.nn.utils.clip_grad_norm_(self.ranking_model.parameters(), abs(self.grad_clip))

                        # this will be executed every other iteration (in order to allow for reward signals)
                        if update:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # h needs to be detached here to avoid in-place modification errors
                            # note for ensembles, this redirects to the children models
                            self.ranking_model.h_prev = self.ranking_model.h_prev.detach()
                            if self.schedulers:
                                for scheduler in self.schedulers:
                                    try:
                                        scheduler.step()
                                    except TypeError:
                                        scheduler.step(loss_critic)
                            self.loss_accum = torch.tensor(0)
                        else:
                            self.loss_accum = torch.tensor(0)
                            
                    if self.control_iter % 2 == 0 or self.online * 10 % 10 == 5:
                        self.old_state_action = [x, edge_index, edge_attr, edge_index_current, edge_attr_current, h_prev_bu, logp, v_score, 
                                                 chosen_nodes, nodes if self.zero_outsample_logits else None]
                            
                else:
                    self.replay_buffer.add((x.cpu(), self.edge_current[0].cpu(), self.edge_current[1].cpu(), edge_accumulate, y.cpu()), \
                                           chosen_nodes.cpu(), logp.sum().item(), v_score.item(), reward, nodes.cpu() if self.zero_outsample_logits else None)
                            
            else:
                # we can utilize as loss weights one of the node features (e.g. degree-weighted loss)
                if self.index_weight >= 0:
                    self.loss.weight = x[:, self.index_weight]
                # if pos_weight enabled, the loss will try to equilibrate the class imbalance
                if self.pos_weight:
                    count = y.count_nonzero()
                    self.loss.pos_weight = (len(y) - count) / count
                loss = loss + self.loss(y_pred, y)
                # accumulating gradients over a certain period
                update = not self.batch_size or self.control_iter % self.batch_size == 0
                if update:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_value_(self.ranking_model.parameters(), self.grad_clip)
                    elif self.grad_clip < 0:
                        torch.nn.utils.clip_grad_norm_(self.ranking_model.parameters(), abs(self.grad_clip))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # h needs to be detached here to avoid in-place modification errors
                    self.ranking_model.h_prev = self.ranking_model.h_prev.detach()
                    if self.schedulers:
                        for scheduler in self.schedulers:
                            try:
                                scheduler.step()
                            except TypeError:
                                scheduler.step(loss)
                    self.loss_accum = torch.tensor(0)
                else:
                    self.loss_accum = loss
            
            if self.debug_print > 0 and self.control_iter % self.debug_print == 0:
                print(f'On day {self.control_day}:')
                metrics = self.metrics
                print(f'Loss: {loss.item()}, Recall: {metrics.recall(y_pred, y.int(), threshold=0)}')
                if self.ranking_model.deterministic:
                    torch.use_deterministic_algorithms(False)
                    confmat = metrics.confusion_matrix(y_pred, y.int(), threshold=0, num_classes=2).cpu().numpy()
                    torch.use_deterministic_algorithms(True)
                else:
                    confmat = metrics.confusion_matrix(y_pred, y.int(), threshold=0, num_classes=2).cpu().numpy()
                df = self.pd.DataFrame(confmat, index=['Actual Neg', 'Actual Pos'], columns=['Pred Neg', 'Pred Pos'])
                print(df)
                print('')

            self.loss_log = loss.item()
            if self.rl_sampler:
                return chosen_nodes.cpu().numpy()
                
        else:
            # which_edge: 0 use only aggregates; 1 use only currents; 2 use all edges
            # mask_pred: 0 no mask; 1 mask aggregates (possibly currents if they are the same obj); 2 mask all
            which_edge, mask_pred = self.pred_vars[:2]
            # if provided, mask_pred_mix allows for masking to be enabled/disabled for a tracer/vaxer
            if not self.testing and len(self.pred_vars) > 2:
                mask_pred = self.pred_vars[2]
            with torch.no_grad():
                if which_edge == 0:
                    edge_current = (edge_index, edge_attr)
                elif which_edge == 1:
                    (edge_index, edge_attr) = edge_current = self.edge_current
                elif which_edge == 2:
                    edge_current = self.edge_current
                # force subgraphing on eligible nodes (thus removing non-actives from message passing)
                if mask_pred > 0:
                    node_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
                    node_mask[nodes] = True
                    # print(self.control_day, 'Node mask:', sum(node_mask), f'{len(nodes)=} {len(net)=}')
                    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                    # print('Prev', len(edge_attr), edge_index.shape)
                    edge_index, edge_attr = edge_index[:, edge_mask], edge_attr[edge_mask]
                    # print('Attr', len(edge_attr), edge_index.shape, 'Subgr', len(net.subgraph(nodes).edges), 'Net-nodes', len(net), 'Net-edge', len(net.edges))
                    if which_edge == 2 and mask_pred == 2:
                        edge_current_mask = node_mask[edge_index_current[0]] & node_mask[edge_index_current[1]]
                        edge_current = (edge_current[0][:, edge_current_mask], edge_current[1][edge_current_mask])
                # i_model = -1 for ensemble, while i_model >= 0 is the index of the submodel; if single model, i_model is ignored
                y_pred = self.ranking_model.predict(x, edge_index, edge_attr=edge_attr, edge_current=edge_current, i_model=self.episode-1)

        self.x = x
        # if explaining is enabled for any process, we need to remember the logits to fit the GraphLIME models to the predictions
        if any(self.explaining):
            self.logits = y_pred.detach()      
        return y_pred.detach().cpu().numpy()
    
    def get_features(self, net, net_changed=True):
        """
        Computes and returns node feature vectors for the given network. This can also use `karateclub` for more advanced embeddings (e.g. node2vec), if available.

        Args:
            net (Network): The network for which to compute the node feature vectors.
            net_changed (bool, optional): Whether the network has changed since the last call to this method. Defaults to False.

        Returns:
            numpy.ndarray: A feature vector for the given network.
        """
        num_nodes = self.n_nodes
        # local vars
        measures = self.static_measures
        centr = self.centr
        dynamic_feat = self.dynamic_feat
        known_inf_neigh = self.known_inf_neigh
        # used to store the feature names (will be used to update the `feats_labels` attribute)
        feats_labels = []
        
        ## compute static features, such as centrality measures and random embeddings
        if net_changed or centr is None:
            centr = np.zeros((num_nodes, len(measures)), dtype=np.float32)
            i = 0
            for measure in measures:
                if ':' in measure:
                    emb_type, emb_dim = measure.split(':')
                    emb_dim = int(emb_dim)
                    if not self.feats_labels:
                        feats_labels += [f'{emb_type}{i}' for i in range(emb_dim)]
                    try:
                        # instatiate embedder type from `karateclub`, if it can be located, and fit to network to obtain embeddings
                        embedder = locate(f'karateclub.{emb_type}')(dimensions=emb_dim)
                        # copy is needed here to avoid adding self-loops to the original net
                        embedder.fit(net.copy())
                        embedding = embedder.get_embedding()
                    except TypeError:
                        if self.rand_emb is None or '+' in measure:
                            self.rand_emb = self.rng.random((num_nodes, emb_dim), dtype=np.float32)
                        embedding = self.rand_emb
                    # delete column i to mark that no feature will be put there
                    centr = np.delete(centr, i, axis=1)
                    # concatenate embedding to the centr array
                    centr = np.concatenate((centr, embedding), axis=1)
                else:
                    splits = measure.split('_')
                    weigh_up = self.weigh_up
                    try:
                        factor = float(splits[-1])
                        measure = measure[:measure.index(splits[-1]) - 1]
                    except ValueError:
                        weigh_up = False
                    if not self.feats_labels:
                        feats_labels.append(splits[0])
                    try:
                        centr_f = getattr(nx, measure + '_centrality')
                        centr_dict = centr_f(net)
                    except AttributeError:
                        centr_dict = getattr(nx, measure)(net)
                    except nx.PowerIterationFailedConvergence:
                        centr_dict = centr_f(net, max_iter=1000)
                    centr[:, i] = [centr_dict.get(i, 0) for i in range(num_nodes)]
                    if weigh_up:
                        centr[:, i] *= factor
                    i += 1
            self.centr = centr
                    
        ## finally, update the features tuple
        feats = [centr]
        if self.k_hops:
            feats.append(known_inf_neigh)
            if not self.feats_labels:
                feats_labels += [f'hop_inf{k}' for k in range(self.k_hops)]
        feats.append(dynamic_feat)
        if not self.feats_labels:
            self.feats_labels = feats_labels + ['untested$_{t-1}$', 'pos$_{t-1}$', 'neg$_{t-1}$', 'ever_pos']           
        # final result will be all features concatenated along columns
        return np.concatenate(feats, axis=1)
    
    
class RLAgent(SLAgent):
    """
    A reinforcement learning agent that extends the functionality of the SLAgent class.
    
    Args:
        rl_sampler (str): The name of the RL sampler to use. Default is 'softmax'. This will signal the other methods in SLAgent that `self` is an RL agent.
        **kwargs: Additional keyword arguments to pass to the parent class constructor.
    """
    def __init__(self, rl_sampler='softmax', **kwargs):
        super().__init__(rl_sampler=rl_sampler, **kwargs)
    
    
class RecordAgent(Agent):
    """
    An agent that records the features and labels of a network at each time step. Note that it has a completely different functionality than other Agents,
    not inheriting anything, yet it supports the same API control(), finish(). This is because the other information present on a standard Agent instance is not needed.

    Args:
        static_measures (tuple): A tuple of strings representing the static measures to compute for each node.
        k_hops (int): The number of hops to consider for known infected neighbors.
        record_edges (bool): Whether to record the edges of the network at each time step.
        n_test (int or float): The number of nodes to test for infection at each time step. If a float between 0 and 1, it represents the fraction of nodes to test.
        seed (int): The random seed to use for testing nodes.
        see_all_uninfected (bool): Whether to see all uninfected nodes or only those that are perceived as uninfected.
        sim_id (int): The ID of the simulation.
        set_infected_attr (bool): Whether to set the `node_infected` attribute on the network.

    Attributes:
        static_measures (tuple): A tuple of strings representing the static measures to compute for each node.
        k_hops (int): The number of hops to consider for known infected neighbors.
        n_test (int): The number of nodes to test for infection at each time step.
        see_all_uninfected (bool): Whether to see all uninfected nodes or only those that are perceived as uninfected.
        sim_id (int): The ID of the simulation.
        set_infected_attr (bool): Whether to set the `node_infected` attribute on the network.
        centr (numpy.ndarray): An array of shape (num_nodes, num_static_measures) containing the computed static measures for each node.
        dynamic_feat (numpy.ndarray): An array of shape (num_nodes, 4) containing the dynamic features for each node.
        known_inf_neigh (numpy.ndarray): An array of shape (num_nodes, k_hops) containing the known infected neighbors for each node.
        rand_emb (numpy.ndarray): An array of shape (num_nodes, emb_dim) containing random node embeddings.
        feats_labels (list): A list of length num_features containing the names of the node features.
        xs (list): A list of length num_time_steps containing the features at each time step.
        ys (list): A list of length num_time_steps containing the labels at each time step.
        edges (list): A list of length num_time_steps containing the edges at each time step, if `record_edges` is True.
        rng (numpy.random.Generator): A random number generator using the supplied seed.

    Methods:
        control(net, control_iter, control_day, initial_known_ids, net_changed): Computes the features and labels for the current time step.
        get_features(net, net_changed): Computes the features for the current time step.
        finish(total_inf, args, sim_id): Stores the features, labels, and edges (if record_edges is True) in the shared dictionary.
    """    
    def __init__(self, static_measures=('degree',), k_hops=2, record_edges=False, n_test=10, seed=None, see_all_uninfected=True, sim_id=0, 
                 set_infected_attr=False, **kwargs):
        self.static_measures = static_measures
        self.k_hops = k_hops
        self.n_test = n_test
        self.see_all_uninfected = see_all_uninfected
        self.sim_id = sim_id
        self.set_infected_attr = set_infected_attr
        self.centr = None
        self.dynamic_feat = None
        self.known_inf_neigh = None
        self.rand_emb = None
        self.feats_labels = None
        self.xs = []
        self.ys = []
        self.edges = [] if record_edges else None
        self.rng = np.random.default_rng(seed)
        
    def control(self, net, control_day=0, initial_known_ids=(), net_changed=True, missed_days=0):
        """
        Records the features, labels, and edges for the current time step.

        Args:
            net (Network): The network to record for.
            control_day (int): The current day of the simulation. Unused argument.
            initial_known_ids (iterable): The IDs of the initially known infected nodes. Unused argument.
            net_changed (bool): Whether the network has changed since the last time step.
            missed_days (int): The number of days since the last control action. Unused argument.

        Returns:
            None
        """
        self.xs.append(self.get_features(net, net_changed))
         # set the `node_infected` attribute if not already set on the network, based on the perceived uninfected states and the node states
        if self.set_infected_attr or not hasattr(net, 'node_infected'):
            self.set_infected_attr = True
            node_states = net.node_states
            perceived_uninf = net.UNINF_STATES if self.see_all_uninfected and hasattr(net, 'UNINF_STATES') else PERCEIVED_UNINF_STATES
            net.node_infected = [s not in perceived_uninf for s in node_states]
        ys = net.node_infected
        # occasionally, `node_infected` may be larger than the actual number of nodes to allow for the node IDs to be discontinuous
        # this means that one needs to select the active nodes from `node_infected`
        self.ys.append(list(ys if len(net) == len(ys) else map(ys.__getitem__, self.all_nodes)))
        if self.edges is not None:
            # we accumulate the edges only if network has changed, otherwise we put a `None` placeholder to avoid memory overhead
            self.edges.append(list(net.to_directed().edges.data("weight", default=1.)) if net_changed else None)
    
    def finish(self, total_inf, args):
        """
        Finish the simulation and store the final state of the RecordAgent (i.e. all features, labels and edges).

        Args:
            total_inf (int): The total number of infected nodes in the simulation. Unused argument.
            args (argparse.Namespace): Command-line arguments. These should contain a dictionary called `shared` where the information will be stored.

        Returns:
            None
        """
        args.shared[self.sim_id] = (self.xs, self.ys, self.edges)

    def get_features(self, net, net_changed=True):
        """
        Computes and returns node feature vectors for the given network. This can also use `karateclub` for more advanced embeddings (e.g. node2vec), if available.

        Args:
            net (Network): The network for which to compute the node feature vectors.
            net_changed (bool, optional): Whether the network has changed since the last call to this method. Defaults to False.

        Returns:
            numpy.ndarray: A feature vector for the given network.
        """
        num_nodes = len(net)
        # local vars
        measures = self.static_measures
        centr = self.centr
        dynamic_feat = self.dynamic_feat
        known_inf_neigh = self.known_inf_neigh
        # used to store the feature names (will be used to update the `feats_labels` attribute)
        feats_labels = []

        ## compute static features, such as centrality measures and random embeddings
        if net_changed or centr is None:
            centr = np.zeros((num_nodes, len(measures)), dtype=np.float32)
            i = 0
            for measure in measures:
                if ':' in measure:
                    emb_type, emb_dim = measure.split(':')
                    emb_dim = int(emb_dim)
                    if not self.feats_labels:
                        feats_labels += [f'{emb_type}{i}' for i in range(emb_dim)]
                    try:
                        # instatiate embedder type from `karateclub`, if it can be located, and fit to network to obtain embeddings
                        embedder = locate(f'karateclub.{emb_type}')(dimensions=emb_dim)
                        # copy is needed here to avoid adding self-loops to the original net
                        embedder.fit(net.copy())
                        embedding = embedder.get_embedding()
                    except TypeError:
                        if self.rand_emb is None or '+' in measure:
                            self.rand_emb = self.rng.random((num_nodes, emb_dim), dtype=np.float32)
                        embedding = self.rand_emb
                    # delete column i to mark that no feature will be put there
                    centr = np.delete(centr, i, axis=1)
                    # concatenate embedding to the centr array
                    centr = np.concatenate((centr, embedding), axis=1)
                else:
                    splits = measure.split('_')
                    weigh_up = self.weigh_up
                    try:
                        factor = float(splits[-1])
                        measure = measure[:measure.index(splits[-1]) - 1]
                    except ValueError:
                        weigh_up = False
                    if not self.feats_labels:
                        feats_labels.append(splits[0])
                    try:
                        centr_f = getattr(nx, measure + '_centrality')
                        centr_dict = centr_f(net)
                    except AttributeError:
                        centr_dict = getattr(nx, measure)(net)
                    except nx.PowerIterationFailedConvergence:
                        centr_dict = centr_f(net, max_iter=1000)
                    centr[:, i] = [centr_dict.get(i, 0) for i in range(num_nodes)]
                    if weigh_up:
                        centr[:, i] *= factor
                    i += 1
            self.centr = centr

        ## update dynamic features, such as the number of known infected neighbors
        if dynamic_feat is None:
            self.all_nodes = list(net)
            dynamic_feat = self.dynamic_feat = np.zeros((num_nodes, 4), dtype=np.float32)
            if self.k_hops:
                known_inf_neigh = self.known_inf_neigh = np.zeros((num_nodes, self.k_hops), dtype=np.float32)
        else:
            # clear current pos/neg tested marker in the dynamic features
            dynamic_feat[:, 1:3] = 0
        # mark all as untested for the current timestamp
        dynamic_feat[:, 0] = 1
        if self.n_test < 1: 
            self.n_test = int(self.n_test * num_nodes)
        # perform dynamic features update based on `n_test` nodes tested at random
        tested =  self.rng.choice(net.node_list, self.n_test, replace=False)
        node_infected = net.node_infected
        for nid in tested:
            dynamic_feat_nid = dynamic_feat[nid]
            # mark person as getting tested in dynamic feat
            dynamic_feat_nid[0] = 0
            if node_infected[nid]:
                # flip both current positive, and the historical positive positions
                dynamic_feat_nid[1] = dynamic_feat_nid[-1] = 1
                if self.k_hops:
                    # accumuiate all neighbors of 'nodes' that are WITHIN k-hops away
                    visited = {nid}
                    # this keeps track of nodes at each depth k (exclusively)
                    k_nbrs = {nid}
                    for k in range(1, self.k_hops + 1):
                        # nodes at depth k-1 starting from nid
                        prev_k_nbrs = k_nbrs.copy()
                        for bfs_src in prev_k_nbrs:
                            k_nbrs.remove(bfs_src)
                            # neighbors at depth k starting from nid, with shortest-path passing through bfs_src
                            for k_nbr in net[bfs_src]:
                                if k_nbr not in visited:
                                    k_nbrs.add(k_nbr)
                                    visited.add(k_nbr)
                                    # increment by one the entry of the depth k neighbor 
                                    # (in its k-1 position, as indexes start from 0)
                                    known_inf_neigh[k_nbr][k - 1] += 1
            else:
                dynamic_feat_nid[2] = 1
           
        ## finally, update the features tuple
        feats = [centr]
        if self.k_hops:
            feats.append(known_inf_neigh)
            if not self.feats_labels:
                feats_labels += [f'hop_inf{k}' for k in range(self.k_hops)]
        feats.append(dynamic_feat)
        if not self.feats_labels:
            self.feats_labels = feats_labels + ['untested$_{t-1}$', 'pos$_{t-1}$', 'neg$_{t-1}$', 'ever_pos']
        ## final result will be all features concatenated along columns
        return np.concatenate(feats, axis=1)
    
    
AGENT_TYPE = {subcls.__name__.partition('Ag')[0].lower(): subcls for subcls in Agent.get_subclasses()}