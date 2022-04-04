import itertools
import copy
import heapq
import random
import numpy as np
import pandas as pd
import networkx as nx
from pydoc import locate
from collections import deque, defaultdict, Counter
from lib.general_utils import tqdm_redirect


UNACCEPTED_FOR_TEST_TRACE = {'H', 'D'}
PERCEIVED_UNINF_STATES = {'S', 'E', 'R'}


class GeometricDataset(list):
    """
    To be used as a PYG Dataset
    """
    def __init__(self, *args, name='Infect', num_node_features=None, num_classes=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.num_node_features = num_node_features if num_node_features is not None else self[0].num_node_features
        self.num_classes = num_classes
        
        
class ReplayBuffer:
    """
    Replay buffer for MC RL algorithms
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logp = []
        self.values = []
        self.rewards = deque()
        self.n_samples = 0
        
    def add(self, *entry):
        self.n_samples += 1
        for i, lst in enumerate(('states', 'actions', 'logp', 'values', 'rewards')):
            self.__dict__[lst].append(entry[i])
        
    def shift_and_discount_rewards(self, last_reward=0, gamma=.99, lamda=.97, reward_scale=1):
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
        self.rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        # replace original values with normalized GAE estimates
        self.values = (adv - adv.mean()) / (adv.std() + 1e-7)

        
    def sample(self, batch_size=32):
        # values are discounted advantages, while rewards are discounted returns
        zipped = (self.states, self.actions, self.logp, self.values, self.rewards)
        for start in range(0, self.n_samples, batch_size):
            idx = range(start, min(start + batch_size, self.n_samples), 1)
            yield [list(map(lst.__getitem__, idx)) for lst in zipped]
    
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logp[:]
        del self.values[:]
        del self.rewards[:]


class Agent:
       
    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass
        
        
    @staticmethod
    def from_dict(typ='centrality', **kwargs):
        return AGENT_TYPE[typ](**kwargs)
    
        
    def __init__(self, tester=True, tracer=False, n_test=10, n_trace=10, keep_pos_until=6, keep_neg_until=3, trace_pos_history=True, trace_ignore_neg_history=False, k_hops=0, seed=None, see_all_uninfected=True, debug_print=0, episode=0, epsilon=0, **kwargs):
        self.n_test = n_test
        self.n_trace = n_trace
        # we'd like to know the maximum number of nodes that will be chosen for either testing or tracing for batching purposes
        self.n_max = max(n_test if tester else 0, n_trace if tracer else 0)
        self.trace_pos_history = trace_pos_history
        self.trace_ignore_neg_history = trace_ignore_neg_history
        self.see_all_uninfected = see_all_uninfected
        self.debug_print = debug_print
        self.episode = episode
        self.eps = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tester = (lambda net, nodes, size: self.control_test(net, nodes, size)) if tester \
                        else (lambda net, nodes, size: self.rng.choice(nodes, size, replace=False))
        self.tracer = (lambda net, nodes, size: self.control_trace(net, nodes, size)) if tracer \
                        else (lambda net, nodes, size: self.rng.choice(np.unique(nodes), size, replace=False))
        self.keep_pos_until = keep_pos_until if keep_pos_until >= 0 else 1e9
        self.keep_neg_until = keep_neg_until if keep_neg_until >= 0 else 1e9
        # number of pos/neg added at each control_iter
        self.pos_added = deque()
        self.neg_added = deque()
        # important negatives and positives history
        self.neg = deque()
        self.pos = deque()
        # whether the network has changed since last update
        self.net_changed = True
        # holds a list of all nodes
        self.all_nodes = None
        # dynamic node features -> (untested, pos, neg, past_pos)
        self.dynamic_feat = None
        # k-hops infected features
        self.k_hops = k_hops
        self.known_inf_neigh = None
        # variable utilized by SL and RL agents
        self.scorer_type = 0
        
        
    def control(self, net, control_iter=0, initial_known_ids=(), net_changed=True):
        # no testing means no possibility of isolating
        if self.n_test <= 0:
            return [], []
        # remembering control_iter is useful for debugging purposes
        self.control_iter = control_iter
        
        known_inf_neigh = self.known_inf_neigh
        dynamic_feat = self.dynamic_feat
        if dynamic_feat is None:
            max_len = len(net)
            self.all_nodes = list(net)
            dynamic_feat = self.dynamic_feat = np.zeros((max_len, 4), dtype=np.float32)
            if self.k_hops:
                known_inf_neigh = self.known_inf_neigh = np.zeros((max_len, self.k_hops), dtype=np.float32)
        else:
            # clear current pos/neg tested marker in the dynamic features
            dynamic_feat[:, 1:3] = 0
        # mark all as untested for the current timestamp
        dynamic_feat[:, 0] = 1

        ### IMPORTANT: The network is assumed to have a node_states and a node_traced field
        # if the agent is SL/RL, the train network also needs to have a node_infected field
        node_states = net.node_states
        node_traced = net.node_traced
        neg = self.neg
        pos = self.pos
        # convert n_test and n_trace to absolutes if percentages selected
        if 0 < self.n_test < 1:
            self.n_test *= len(net)
        if 0 < self.n_trace < 1:
            self.n_trace *= len(net)
        # mark whether the network has changed since last call
        self.net_changed = net_changed
        
        # timestamp for dequeuing pos/neg that were tested too further ago
        pos_from = control_iter - self.keep_pos_until - 1
        neg_from = control_iter - self.keep_neg_until - 1
        # deque from positives those that were tested a long time ago
        if pos_from >= 0:
            n_add = self.pos_added.popleft()
            for i in range(n_add):
                pos.popleft()
        # deque from negatives those that were tested a long time ago
        if neg_from >= 0:
            n_remove = self.neg_added.popleft()
            for i in range(n_remove):
                neg.popleft()
                        
        # filter 'acitve' nodes (i.e. NOT hospitalized/dead) that are NOT currently traced
        # NOTE: this currently means isolated people do not get further testing
        active = list(filter(lambda nid: node_states[nid] not in UNACCEPTED_FOR_TEST_TRACE 
                             and not node_traced[nid] and nid not in initial_known_ids, net.node_list))
        # population that is valid for testing: i.e. active - recently-tested negatives
        valid_for_test = np.setdiff1d(active, neg, assume_unique=True)
        
        # TO-RETURN: list of newly tested positives
        new_pos = []
        pos_count = neg_count = 0
        perceived_uninf = net.UNINF_STATES if self.see_all_uninfected else PERCEIVED_UNINF_STATES

        # verify if sampling is actually possible in terms of population size, otherwise mark all nodes left as valid for tesing
        if self.n_test < len(valid_for_test):
            # logic for making the agent aware of the initial_known_ids as positives
            if initial_known_ids is not None and len(initial_known_ids):
                for nid in initial_known_ids:
                    dynamic_feat_nid = dynamic_feat[nid]
                    # mark person as getting tested in dynamic feat
                    dynamic_feat_nid[0] = 0
                    pos.append(nid)
                    new_pos.append(nid)
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
                                for k_nbr in net[bfs_src]:
                                    if k_nbr not in visited:
                                        k_nbrs.add(k_nbr)
                                        visited.add(k_nbr)
                                        # increment by one the entry of the depth k neighbor 
                                        # (in its k-1 position, as indexes start from 0)
                                        known_inf_neigh[k_nbr][k - 1] += 1
                    
            # test only people that have not been tested negative recently
            tested = self.tester(net, valid_for_test, size=self.n_test)
            # print(tested)
            trace_possible = True
        else:
            tested = valid_for_test
            trace_possible = False
                                
        # # extend list of tested with initially known infected
        # if initial_known_ids is not None and len(initial_known_ids):
        #     tested = np.append(tested, initial_known_ids, axis=0)        

        # partition tested into positives and negatives
        for nid in tested:
            dynamic_feat_nid = dynamic_feat[nid]
            # mark person as getting tested in dynamic feat
            dynamic_feat_nid[0] = 0
            if node_states[nid] not in perceived_uninf:
                pos.append(nid)
                new_pos.append(nid)
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
                            for k_nbr in net[bfs_src]:
                                if k_nbr not in visited:
                                    k_nbrs.add(k_nbr)
                                    visited.add(k_nbr)
                                    # increment by one the entry of the depth k neighbor 
                                    # (in its k-1 position, as indexes start from 0)
                                    known_inf_neigh[k_nbr][k - 1] += 1
            else:
                neg.append(nid)
                neg_count += 1
                dynamic_feat_nid[2] = 1

        # update the count of neg added for the current timestamp
        self.pos_added.append(pos_count)
        self.neg_added.append(neg_count)
        
        # TO-RETURN: list of newly traced individuals
        traced = []
        if trace_possible and self.n_trace > 0:
            trace_nbrs_of = pos if self.trace_pos_history else new_pos
            trace_ignore = set(tested) if self.trace_ignore_neg_history else set(neg) | set(new_pos)
            # get neighbors of positives that are active, and that are NOT part of trace_ignore
            # this list MAY contain DUPLICATES, which some controllers may leverage
            valid_for_trace = [neigh for nid in trace_nbrs_of for neigh in net[nid]
                               if not node_traced[neigh] and node_states[neigh] not in UNACCEPTED_FOR_TEST_TRACE 
                               and neigh not in trace_ignore]
            set_valid_for_trace = set(valid_for_trace)
            if self.n_trace < len(set_valid_for_trace):
                traced = self.tracer(net, valid_for_trace, size=self.n_trace)
            else:
                traced = set_valid_for_trace
        
        return new_pos, traced
            
            
    def control_test(self, net, nodes, size=10, **kwargs):
        """
        By default, if a custom control strategy is selected here (tester=True), the logic will be controlled through 'control_both'
        However, subclasses can extend this to offer special treatment for test control.
        """
        return self.control_both(net, nodes, size, **kwargs)
        
        
    def control_trace(self, net, nodes, size=10, **kwargs):
        """
        By default, if a custom control strategy is selected here (tracer=True), the logic will be controlled through 'control_both'
        However, subclasses can extend this to offer special treatment for tracing control.
        """
        return self.control_both(net, nodes, size, **kwargs)
        
        
    def control_both(self, net, nodes, size=10, **kwargs):
        """
        This method provides a common logic for controlling both testing and tracing, and should be overridden if no special behavior
        is needed for either testing or tracing.
        """
        raise NotImplementedError('This method was not implemented for this type of Agent. It may be the case that this Agent cannot control both testing and tracing.')
        

class MixAgent(Agent):
    
    def __init__(self, test_type, trace_type, **kwargs):
        self.test_agent = AGENT_TYPE[test_type](**kwargs)
        self.trace_agent = AGENT_TYPE[trace_type](**kwargs)
        super().__init__(**kwargs)
        
        
    def control_test(self, net, nodes, size=10, **kwargs):
        self.test_agent.net_changed = self.net_changed
        return self.test_agent.control_test(net, nodes, size, **kwargs)
        
        
    def control_trace(self, net, nodes, size=10, **kwargs):
        self.trace_agent.net_changed = self.net_changed
        return self.trace_agent.control_trace(net, nodes, size, **kwargs)
    
    
class FrequencyAgent(Agent):
    
    def control_trace(self, net, nodes, size=10, **kwargs):
        return [entry[0] for entry in Counter(nodes).most_common(size)]
    
    
class MeasureAgent(Agent):
    
    def __init__(self, **kwargs):
        self.computed_measures = None
        super().__init__(**kwargs)
        
        
    def control_both(self, net, nodes, size=10, **kwargs):
        # compute new measures if needed
        if self.update_condition():
            self.computed_measures = self.compute_measures(net, nodes, size)
        # allow for special behavior to occur (i.e. a RL agent in learning mode, with lr > 0 and rl_sampler not None)
        # in this case, self.computed_measures will directly hold the nodes that should be selected by the agent in this iteration
        if self.scorer_type == 2:
            return self.computed_measures
            
        rankers = self.computed_measures
        # init priority queue to return best scored of len = 'size'
        pq = []
        for nid in set(nodes):
            if len(pq) < size:
                heapq.heappush(pq, (rankers[nid], nid))
            else:
                heapq.heappushpop(pq, (rankers[nid], nid))
#         print('DAY ' + str(self.control_iter))
#         print('-----------')
#         print('Overall measures:', dict(self.computed_measures), '-----------')
#         print('Negatives on day ' + str(self.control_iter - 1), self.neg, '-----------')
#         print('Chosen node states:')
#         print([(nid, net.node_states[nid], net.node_traced[nid]) for nid in nodes])
#         print('-----------', 'PQ:', pq)
#         print('==================')

        # return the nid corresponding to the most important entries in pq
        return [entry[1] for entry in pq]
    
    
    def update_condition(self):
        """
        Classes can override this if the measurement which guides the nodes ranking only needs sporadic updates
        """
        return True
    
        
    def compute_measures(self, net, nodes, size=10):
        raise NotImplementedError('A measurement to compare nodes needs to be implemented.')
    
        
class CentralityAgent(MeasureAgent):
    
    def __init__(self, measure='degree', **kwargs):
        self.measure = measure
        super().__init__(**kwargs)
        
        
    def update_condition(self):
        return self.net_changed or self.computed_measures is None
        
        
    def compute_measures(self, net, nodes, size=10):
        """
        For Centrality-based agents, we will use the networkx API, and thus most of the metrics will be computed from functions
        that do not support restricting the computation to node subsets.
        """
        try:
            centr_f = getattr(nx, self.measure + '_centrality')
            return centr_f(net)
        except AttributeError:
            return getattr(nx, self.measure)(net)
        except nx.PowerIterationFailedConvergence:
            return centr_f(net, max_iter=1000)
        
        
class WeightAgent(CentralityAgent):
    
    def compute_measures(self, net, nodes, size=10):
        return dict(net.degree(weight='weight'))
    
    
class NeighborsAgent(MeasureAgent):
    
    def compute_measures(self, net, nodes, size=10):
        return self.known_inf_neigh.tolist()
    
    
class SLAgent(MeasureAgent):
            
    def __init__(self, ranking_model=None, target_model=None, static_measures=('degree','eigenvector'), gpu=False, lr=0, mark_delay_same_edges=False, index_weight=-1, pos_weight=False, need_dynamic_compute=False, optimizer='Adam', scheduler=None, grad_clip=0, rl_sampler=None, online=True, rl_args=(.99, .97, 1, 'ppo', -.2, .5, .01, 0, 1, 5), batch_size=0, epochs=1, eps=.5, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.static_measures = static_measures
        self.need_dynamic_compute = need_dynamic_compute
        self.mark_delay_same_edges = mark_delay_same_edges
        self.grad_clip = grad_clip
        self.online = online
        self.batch_size = batch_size
        self.epochs = epochs
        self.dist_over_sample = kwargs.get('dist_over_sample', True)
        self.logp_from_score = kwargs.get('logp_from_score', True)
        self.logpold_from_score = kwargs.get('logpold_from_score', True)
        
        # local import of torch
        torch = self.torch = __import__('torch')
        self.gpu = gpu and torch.cuda.is_available()
        self.lr = lr
        self.rl_sampler = rl_sampler
        self.total_reward = 0
        
        # memory: static measures
        self.centr = None
        # memory: cummulative edges and edge attrs
        self.edge_index = None
        self.edge_attr = None
        self.edge_current = None
        
        # only importing iff debug prints is enabled
        if self.debug_print:
            self.metrics = __import__('torchmetrics').functional
            
        is_target_exist = target_model is not None
        
        if not ranking_model:
            from lib.rank_model import RankingModel
            # number of dynamics features are 4 (for self.dynamic_feat) + k_hops (for self.known_inf_neigh)
            input_features = 4 + self.k_hops
            for measure in static_measures:
                try:
                    input_features += int(measure.split(':')[1])
                except IndexError:
                    input_features += 1
            ranking_model = RankingModel(input_features, 1, torch_seed=self.seed)
        else:
            # make sure previous iteration's h_prev are never utilized
            ranking_model.h_prev = None
            if is_target_exist:
                target_model.h_prev = None
           
        ## for debugging purposes; this does NOT work with multiprocessing and should be removed soon
        if debug:
            ranking_model.agent = self
            if is_target_exist:
                target_model.agent = self
        
        device_type = ranking_model.device.type
        if self.gpu:
            if device_type == 'cpu':
                ranking_model = ranking_model.cuda()
                if self.rl_sampler and is_target_exist:
                    target_model = target_model.cuda()
        elif device_type != 'cpu':
            ranking_model = ranking_model.cpu()
            if self.rl_sampler and is_target_exist:
                target_model = target_model.cpu()
                
        if lr:
            ranking_model.train()
            self.optimizer = getattr(torch.optim, optimizer)(ranking_model.parameters(), lr=lr, **kwargs.get('optimizer_kwargs', {})) \
                                if isinstance(optimizer, str) else optimizer
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(self.optimizer, *kwargs.get('scheduler_args', (100))) \
                                if isinstance(scheduler, str) else scheduler
            self.loss = kwargs.get('loss', torch.nn.BCEWithLogitsLoss())
            if rl_sampler:
                self.scorer_type = 2
                self.gamma, self.lamda, self.reward_scale, self.actor_loss, self.clip_ratio, self.value_coeff,\
                    self.entropy_coeff, self.ce_coeff, self.eligibility, self.target_update = rl_args
                if online:
                    # accumulating gradients over a certain period; since RL updates happen every 2 steps, batch_size must be uneven
                    self.batch_size = batch_size // 2 * 2 + 1
                    # eligibility trace of the loss gradients
                    self.trace = [torch.zeros_like(p.data, requires_grad=False) for p in ranking_model.parameters()]
                    # logic to delay the update of the target value network
                    if is_target_exist and self.episode and self.episode % self.target_update == 0:
                        print('Changing target model...')
                        target_model.load_state_dict(ranking_model.state_dict())
                else:
                    # local import of PYG needed to construct PYG batches
                    self.pyg_data = __import__('torch_geometric').data
                    # when not online, a replay buffer will be needed
                    self.replay_buffer = ReplayBuffer()
            else:
                # accumulating gradients over a certain period
                self.batch_size = batch_size
                self.index_weight = index_weight
                self.pos_weight = pos_weight
        # if no lr has been given, then assume the model should be put into eval mode
        elif ranking_model.training:
            ranking_model.eval()
        # finally, assign the models as instance variables to the Agent
        self.ranking_model = ranking_model
        self.target_model = target_model
        
        
    def update_parameters(self, last_reward=0):
        torch = self.torch
        self.replay_buffer.shift_and_discount_rewards(last_reward, self.gamma, self.lamda, self.reward_scale)
        device = self.ranking_model.device
        
        if self.epochs < 0:
            outer_loop = lambda: tqdm_redirect(range(-self.epochs), total=-self.epochs)
            batch_loop = lambda: self.replay_buffer.sample(batch_size=self.batch_size)
            inner_loop = 1
        elif self.epochs > 0:
            outer_loop = lambda: range(1)
            batch_loop = lambda: tqdm_redirect(self.replay_buffer.sample(batch_size=self.batch_size), total=self.replay_buffer.n_samples//self.batch_size)
            inner_loop = self.epochs
        else:
            raise ValueError('The value of "epochs" cannot be 0')           
        
        for o in outer_loop():
            edge_index = None
            edge_attr = None
            self.ranking_model.h_prev = None
            for states, actions, logp_old, values, rewards in batch_loop():
                adv = torch.tensor(values, device=device)
                returns = torch.tensor(rewards, device=device)
                logp_old = torch.tensor(logp_old, device=device)
                actions = torch.stack(actions)
                # print(self.ranking_model.info.hidden[0].nn[-1].bias)

                # establish the true batch_size (taking into consideration the last batch which may be incomplete)
                if len(states) == self.batch_size:
                    batch_size = self.batch_size
                else:
                    batch_size = len(states)
                    index_keep = len(self.all_nodes) * (self.batch_size - batch_size)
                    self.ranking_model.h_prev = self.ranking_model.h_prev[index_keep:]

                x, edge_index_current, edge_attr_current, edge_accumulate, y = zip(*states)
                if edge_index is None:
                    # we want edge_index and edge_attr to be lists instead of tuples to allow for item assignment when accumulating the temporal graph
                    edge_index = list(edge_index_current)
                    edge_attr = list(edge_attr_current)
                else:
                    for i, accum in enumerate(edge_accumulate):
                        if accum:
                            edge_index[i] = torch.cat((edge_index[i], edge_index_current[i]), dim=1)
                            # increase by 1 the time delay of all other timestamps (Note, the current one is yet to be appended)
                            edge_attr[i][:, 1] += 1
                            edge_attr[i] = torch.cat((edge_attr[i], edge_attr_current[i]), dim=0)

                ## PYG batching
                # note, we pass a tuple of both edge_index and edge_index_current to ensure both get batches as adjancey matrices
                batch = self.pyg_data.Batch.from_data_list([
                    self.pyg_data.Data(x=x[i], edge_index=(edge_index[i], edge_index_current[i]), edge_attr=edge_attr[i], 
                                       edge_attr_current=edge_attr_current[i]) for i in range(batch_size)])
                if self.gpu:
                    batch = batch.to(device)
                    actions = actions.to(device)

                for i in range(inner_loop):
                    loss = torch.tensor(0)
                    # the raw scores in y_pred are actually logits of the policy
                    y_pred, v_score = self.ranking_model(batch.x, batch.edge_index[0], batch.edge_attr, batch_idx=batch.batch, \
                                                         edge_current=(batch.edge_index[1], batch.edge_attr_current), scorer_type=2)
                    y_pred = y_pred.reshape(batch_size, -1)
                    
                    if self.dist_over_sample:
                        scores = y_pred.gather(1, actions)
                        pi, _ = self.score_to_prob(scores)
                        logp = scores.sum(dim=-1)
                    else:
                        pi, _ = self.score_to_prob(y_pred)
                        logp = (y_pred.gather(1, actions) if self.logp_from_score else pi.log_prob(actions.T).T).sum(dim=-1) 
                    # print(f'\nDist: {logp - logp_old} \n vs {y_pred.view(batch_size, -1).gather(1, actions).sum(dim=-1) - logp_old}')

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
                        neg_of_entropy = -pi.entropy().mean()
                        loss = loss + self.entropy_coeff * neg_of_entropy
                    if self.ce_coeff:
                        # multitask learning with PPO + CE loss
                        loss_ce = self.loss(y_pred, torch.stack(y).to(device))
                        loss = loss + self.ce_coeff * loss_ce
                    # backproing through pi.log_prob is non-deterministic on CUDA due to the use of 'scatter_add_cuda'
                    if self.gpu and not self.logp_from_score and self.ranking_model.deterministic:
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
                    if self.scheduler is not None:
                        self.scheduler.step()
   
    
    def score_to_prob(self, scores, nodes=None, size=10):
        """
        nodes - list of nodes to sample from (if None, no sampling is performed)
        """
        torch = self.torch
        if self.rl_sampler == 'greedy':
            pi = torch.distribution.Categorical(logits=scores)
            # when acting greedy, we always need to refer to the scores of the sample nodes (even if dist_over_sample is False)
            nodes_scores = scores if self.dist_over_sample else scores[nodes]
            # random choice or partial sorting of node scores and taking maximum
            sampled_indices = lambda: self.rng.choice(len(nodes), size=size, replace=False) if self.rng.uniform(0, 1) < self.eps \
                                else np.argpartition(nodes_scores.detach().cpu().numpy(), -size)[-size:]
        else:
            if self.rl_sampler == 'softmax':
                probs = torch.nn.functional.softmax(scores / self.eps, dim=0)
            elif self.rl_sampler == 'log':
                probs = torch.exp(torch.nn.functional.log_softmax(scores / self.eps, dim=0))
            elif self.rl_sampler == 'meirom':
                probs = torch.clamp(scores - scores.min() + self.eps, min=0)
                probs = probs / probs.sum()
            elif self.rl_sampler == 'escort':
                # print(self.eps)
                probs = scores.abs() ** (1 / self.eps)
                probs = probs / probs.sum()
            pi = torch.distributions.Categorical(probs=probs)
            # when sampling, we always need to refer to the probs of the sample nodes (even if dist_over_sample is False)            
            nodes_probs = probs if self.dist_over_sample else probs[nodes] + 1e-45
            sampled_indices = lambda: torch.multinomial(nodes_probs, num_samples=size)
        
        if nodes is not None:
            chosen_nodes = torch.tensor([nodes[idx] for idx in sampled_indices()], dtype=int, device=scores.device)
            return pi, chosen_nodes
        return pi, None
        

    def compute_measures(self, net, nodes=None, size=10):
        if nodes is None or len(nodes) == 0:
            nodes = list(net)
        torch = self.torch
        device = self.ranking_model.device
        if self.gpu and self.control_iter % 10 == 0:
            torch.cuda.empty_cache()
        
        # if self.control_iter % 50 == 0:
        #     print(self.ranking_model.f_scorer.model[0].weight)
        
        x = torch.from_numpy(self.get_features(net, self.net_changed)).to(device)
        edge_accumulate = False
        edge_index, edge_attr = self.edge_index, self.edge_attr
        # if the edges have not changed since last time, at least one timepoint happened, and mark_delay_same_edges is disabled, self contains all the info needed
        if edge_index is not None and not self.net_changed and not self.mark_delay_same_edges:
            edge_index_current, edge_attr_current = self.edge_current
        # otherwise, edge_current tensors will need to be created from the supplied (updated) network
        else:
            edges_current_time = torch.tensor(list(net.to_directed().edges.data('weight', default=1.)), device=device)
            # transform list of edges into COO format for torch geometric
            edge_index_current = edges_current_time[:, :2].long().t().contiguous()
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
                edge_index = torch.cat((edge_index, edge_index_current), dim=1)
                # increase by 1 the time delay of all other timestamps (Note, the current one is yet to be appended)
                edge_attr[:, 1] += 1
                edge_attr = torch.cat((edge_attr, edge_attr_current), dim=0)
        # update the memorized temporal edge_index and edge_attr
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_current = (edge_index_current, edge_attr_current)
            
        if self.lr:
            loss = torch.tensor(0)
            # get the predicted infection status of all nodes (and optionally the state score, for scorer_type=2)
            y_pred, v_score = self.ranking_model(x, edge_index, edge_attr, edge_current=self.edge_current, scorer_type=self.scorer_type)
            # get the true infection status of all nodes (i.e. y_true)
            y = torch.tensor(net.node_infected, dtype=float, device=y_pred.device)
            # sometimes node_infected may be larger than the actual number of nodes to allow for nids to be discontinuous
            # this means that one needs to select the active nodes from the node_infected list to get the true list of infections
            if len(x) != len(y):
                y = y[self.all_nodes]
            
            if self.rl_sampler:
                # reward signal corresponding to the 'action' taken in the previous state, arriving in the new state
                reward = -y.count_nonzero().item()
                self.total_reward += reward
                
                # agent.get_action() (env.step will be delayed from this point)
                if self.dist_over_sample:
                    pi, chosen_nodes = self.score_to_prob(y_pred[nodes], nodes=nodes, size=size)
                    logp = y_pred[chosen_nodes]
                else:
                    pi, chosen_nodes = self.score_to_prob(y_pred, nodes=nodes, size=size)
                    logp = y_pred[chosen_nodes] if self.logpold_from_score else pi.log_prob(chosen_nodes)                 
                
                if self.online:
                    if self.control_iter % 2 == 0:
                        self.old_state_action = [logp, v_score, chosen_nodes]
                    else:
                        # old_state remembers the action taken to get to this stage, as well as the previous node and state scores
                        logp_old, v_score_old, chosen_nodes_old = self.old_state_action
                        # for PPO, we want to base logp of the action remembered in old_state rather than the chosen action in the present state
                        logp = y_pred[chosen_nodes_old] if self.dist_over_sample or self.logp_from_score else pi.log_prob(chosen_nodes_old)
                        if self.target_model:
                            with torch.no_grad():
                                _, v_target = self.target_model(x, edge_index, edge_attr, edge_current=self.edge_current, scorer_type=1)
                        else:
                            v_target = v_score.detach()

                        # calculate td error graph and value
                        delta = self.reward_scale * reward  + self.gamma * v_target - v_score_old
                        delta_value = delta.item()
                        if self.clip_ratio:
                            # calculate the actor loss
                            adv = delta_value
                            ratio = torch.exp(torch.clamp(logp - logp_old.detach(), max=50))
                            # allow for the loss to point towards or against logp via the sign of clip_ratio
                            # the default behavior should be: minimize a loss that points against logp, equivalent to maximizing logp
                            direction = np.sign(self.clip_ratio)
                            clip_ratio = torch.tensor(abs(self.clip_ratio))
                            if self.actor_loss == 'ppo':
                                # clip = torch.clamp(ratio, min=1-clip_ratio, max=1+clip_ratio)
                                # raw_objective = torch.min(ratio * adv, clip * adv)
                                raw_objective = adv * (torch.min(ratio, 1 + clip_ratio) \
                                                       if adv >= 0 else torch.max(ratio, 1 - clip_ratio))
                            elif self.actor_loss == 'a2c':
                                raw_objective = logp_old * adv
                            else:
                                raw_objective = logp * adv
                            loss_actor = direction * raw_objective.mean()
                            # print(loss_actor)
                            loss = loss + loss_actor
                        if self.value_coeff:
                            # calculate the critic loss
                            loss_critic = delta ** 2
                            # print('Val:', self.value_coeff * loss_critic)
                            loss = loss + self.value_coeff * loss_critic
                        if self.entropy_coeff:
                            # calculate the -Entropy of the new policy
                            # model should maximize the Entropy which means minimizing -Entropy
                            neg_of_entropy = -pi.entropy()
                            loss = loss + self.entropy_coeff * neg_of_entropy
                        if self.ce_coeff:
                            # multitask learning with PPO + CE loss
                            loss_ce = self.loss(y_pred, y)
                            loss = loss + self.ce_coeff * loss_ce

                        # backproing through pi.log_prob is non-deterministic on CUDA due to the use of 'scatter_add_cuda'
                        if self.gpu and not self.logp_from_score and self.ranking_model.deterministic:
                            torch.use_deterministic_algorithms(False)
                            loss.backward()
                            torch.use_deterministic_algorithms(True)
                        else:
                            loss.backward()
                        if self.eligibility:
                            for idx, p in enumerate(self.ranking_model.parameters()):
                                if p.grad is not None:
                                    self.trace[idx] = self.gamma * self.lamda * self.trace[idx] + p.grad
                                    p.grad = delta_value * self.trace[idx]
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_value_(self.ranking_model.parameters(), self.grad_clip)
                        elif self.grad_clip < 0:
                            torch.nn.utils.clip_grad_norm_(self.ranking_model.parameters(), abs(self.grad_clip))

                        # this will be executed every other iteration (in order to allow for reward signals)
                        if not self.batch_size or self.control_iter % self.batch_size == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            if self.scheduler is not None:
                                self.scheduler.step()
                else:
                    self.replay_buffer.add((x.cpu(), self.edge_current[0].cpu(), self.edge_current[1].cpu(), edge_accumulate, y.cpu()), chosen_nodes.cpu(), \
                                           logp.sum().item(), v_score.item(), reward)
                            
            else:
                # we can utilize as loss weights one of the node features (e.g. degree-weighted loss)
                if self.index_weight >= 0:
                    self.loss.weight = x.detach()[:, self.index_weight]
                # is pos_weight enabled, the loss will try to equilibrate the class imbalance
                if self.pos_weight:
                    count = y.count_nonzero()
                    self.loss.pos_weight = (len(y) - count) / count
                loss = self.loss(y_pred, y)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.ranking_model.parameters(), self.grad_clip)
                elif self.grad_clip < 0:
                    torch.nn.utils.clip_grad_norm_(self.ranking_model.parameters(), abs(self.grad_clip))
            
                if not self.batch_size or self.control_iter % self.batch_size == 0: 
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            
            if self.debug_print > 0 and self.control_iter % self.debug_print == 0:
                print(f'On day {self.control_iter}:')
                metrics = self.metrics
                print(f'Loss: {loss.item()}, Recall: {metrics.recall(y_pred, y.int(), threshold=0)}')
                confmat = metrics.confusion_matrix(y_pred, y.int(), threshold=0, num_classes=2).numpy()
                df = pd.DataFrame(confmat, index=['Actual Neg', 'Actual Pos'], columns=['Pred Neg', 'Pred Pos'])
                print(df)
                print('')
                
            if self.rl_sampler:
                return chosen_nodes.cpu().numpy()
                
        else:
            with torch.no_grad():
                node_mask = torch.zeros(len(net), dtype=torch.bool)
                node_mask[nodes] = True
                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                y_pred = self.ranking_model.predict(x, edge_index[:, edge_mask], edge_attr=edge_attr[edge_mask], i_model=self.episode-1)
                
        return y_pred.detach().cpu().numpy()
    
    
    def get_features(self, net, net_changed=True):
        num_nodes = len(net)
        # local vars
        measures = self.static_measures
        centr = self.centr
        dynamic_feat = self.dynamic_feat
        known_inf_neigh = self.known_inf_neigh
        
        if net_changed or centr is None:
            centr = np.zeros((num_nodes, len(measures)), dtype=np.float32)
            i = 0
            for measure in measures:
                if ':' in measure:
                    emb_type, emb_dim = measure.split(':')
                    emb_dim = int(emb_dim)
                    try:
                        # instatiate embedder from karateclub and fit to network
                        embedder = locate(f'karateclub.{emb_type}')(dimensions=emb_dim)
                        # copy is needed here to avoid adding self-loops to the original net
                        embedder.fit(net.copy())
                        embedding = embedder.get_embedding()
                    except TypeError:
                        embedding = self.rng.random((num_nodes, emb_dim), dtype=np.float32)
                    # delete column i to mark that no feature will be put there
                    centr = np.delete(centr, i, axis=1)
                    # concatenate embedding to the centr array
                    centr = np.concatenate((centr, embedding), axis=1)
                else:
                    try:
                        centr_f = getattr(nx, measure + '_centrality')
                        centr_dict = centr_f(net)
                    except AttributeError:
                        centr_dict = getattr(nx, measure)(net)
                    except nx.PowerIterationFailedConvergence:
                        centr_dict = centr_f(net, max_iter=1000)
                    centr[:, i] = list(centr_dict.values())
                    i += 1
            self.centr = centr
        
        ## compute on-the-fly dynamic_feat and known_inf_neigh iff need_dynamic_update
        # THIS SHOULD THEORETICALLY EXECUTE ONLY WHEN DEBUGGING
        if self.need_dynamic_compute:
            if dynamic_feat is None:
                dynamic_feat = self.dynamic_feat = np.zeros((num_nodes, 4), dtype=np.float32)
                if self.k_hops:
                    known_inf_neigh = self.known_inf_neigh = np.zeros((num_nodes, self.k_hops), dtype=np.float32)
            else:
                # clear current pos/neg tested marker in the dynamic features
                dynamic_feat[:, 1:3] = 0
            # mark all as untested for the current timestamp
            dynamic_feat[:, 0] = 1

            if self.n_test < 1: self.n_test = int(self.n_test * num_nodes)
            tested =  self.rng.choice(net.node_list, self.n_test, replace=False)
            node_states = net.node_states
            perceived_uninf = net.UNINF_STATES if self.see_all_uninfected else PERCEIVED_UNINF_STATES
            for nid in tested:
                dynamic_feat_nid = dynamic_feat[nid]
                # mark person as getting tested in dynamic feat
                dynamic_feat_nid[0] = 0
                if node_states[nid] not in perceived_uninf:
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
        feats.append(dynamic_feat)
                        
        # final result will be all features concatenated along columns
        return np.concatenate(feats, axis=1)
    
    
class RLAgent(SLAgent):
    
    def __init__(self, rl_sampler='softmax', **kwargs):
        super().__init__(rl_sampler=rl_sampler, **kwargs)
    
    
class RecordAgent(Agent):
    
    def __init__(self, static_measures=('degree','eigenvector'), k_hops=2, record_edges=False, n_test=10, seed=None, see_all_uninfected=True, **kwargs):
        self.static_measures = static_measures
        self.k_hops = k_hops
        self.n_test = n_test
        self.see_all_uninfected = see_all_uninfected
        self.centr = None
        self.dynamic_feat = None
        self.known_inf_neigh = None
        self.xs = []
        self.ys = []
        self.edges = [] if record_edges else None
        self.rng = np.random.default_rng(seed)
        
    
    def control(self, net, control_iter=0, initial_known_ids=(), net_changed=True):
        self.xs.append(self.get_features(net, net_changed))
        ys = net.node_infected
        # sometimes node_infected may be larger than the actual number of nodes to allow for nids to be discontinuous
        # this means that one needs to select the active nodes from node_infected
        self.ys.append(list(ys if len(net) == len(ys) else map(ys.__getitem__, self.all_nodes)))
        if self.edges is not None:
            # we accumulate the edges only if network has changed, otherwise we put a None placeholder
            self.edges.append(list(net.to_directed().edges.data("weight", default=1.)) if net_changed else None)
            
    
    def get_features(self, net, net_changed=False):
        num_nodes = len(net)
        # local vars
        measures = self.static_measures
        centr = self.centr
        dynamic_feat = self.dynamic_feat
        known_inf_neigh = self.known_inf_neigh
        
        if net_changed or centr is None:
            centr = np.zeros((num_nodes, len(measures)), dtype=np.float32)
            i = 0
            for measure in measures:
                if ':' in measure:
                    emb_type, emb_dim = measure.split(':')
                    emb_dim = int(emb_dim)
                    try:
                        # instatiate embedder from karateclub and fit to network
                        embedder = locate(f'karateclub.{emb_type}')(dimensions=emb_dim)
                        # copy is needed here to avoid adding self-loops to the original net
                        embedder.fit(net.copy())
                        embedding = embedder.get_embedding()
                    except TypeError:
                        embedding = self.rng.random((num_nodes, emb_dim), dtype=np.float32)
                    # delete column i to mark that no feature will be put there
                    centr = np.delete(centr, i, axis=1)
                    # concatenate embedding to the centr array
                    centr = np.concatenate((centr, embedding), axis=1)
                else:
                    try:
                        centr_dict = getattr(nx, measure + '_centrality')(net)
                    except AttributeError:
                        centr_dict = getattr(nx, measure)(net)
                    centr[:, i] = list(centr_dict.values())
                    i += 1
            self.centr = centr

        ## prepare dynamic features for update
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
        ## perform dynamic features update based on n_test nodes
        if self.n_test < 1: self.n_test = int(self.n_test * num_nodes)
        tested =  self.rng.choice(net.node_list, self.n_test, replace=False)
        node_states = net.node_states
        perceived_uninf = net.UNINF_STATES if self.see_all_uninfected else PERCEIVED_UNINF_STATES
        for nid in tested:
            dynamic_feat_nid = dynamic_feat[nid]
            # mark person as getting tested in dynamic feat
            dynamic_feat_nid[0] = 0
            if node_states[nid] not in perceived_uninf:
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
        feats.append(dynamic_feat)
            
        # final result will be all features concatenated along columns
        return np.concatenate(feats, axis=1)
    
    
def get_dataset_mp(mp_result, sample_every=1, mark_delay_same_edges=False):
    """
    mp_result is assumed to be a hierarchical representation of gathered information across simulations.
    Its dimensions are num_iters x 3(i.e. xs, ys, edges) x num_events(i.e. time) x num_nodes x num_features
    """
    populate_dataset = GeometricDataset(num_node_features=mp_result[0][0][0].shape[1], num_classes=2)
    lens = []
    for key, entry in mp_result.items():
        # occasionally, the mp_result may hold entries that are not to be used for constructing a PYG dataset
        # for convenience, PYG dataset entries will use keys that are integers (corresponding to the iteration number)
        if isinstance(key, int):
            get_dataset(None, populate_dataset, sample_every, mark_delay_same_edges, *entry)
            lens.append(len(populate_dataset))
    return populate_dataset, lens


def get_dataset(record_agent=None, populate_dataset=None, sample_every=1, mark_delay_same_edges=False, xs=None, ys=None, edges_over_time=None):
    """
    Constructs a PYG-compatible dataset from either a record_agent instance OR externally supplied xs, ys and edges
    """
    # local imports to avoid global imports of torch in multiprocessing (memory issues)
    import torch
    from torch_geometric.data import Data
    # retrieve either from args or from self vars
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
            # a dumy exception to escape the duplication of prev timestamp's edges with different timedelays
            # for the case in which edges_over_time[t] is None AND mark_delay_same_edges is disabled
            elif not mark_delay_same_edges:
                raise HandleDisabledMarkDelay
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
        # - edges_over_time[t] is None AND mark_delay_same_edges was disabled, 
        # in which case we want to utilize previous timestamp's edges with NO duplication for timedelay purposes
        except HandleDisabledMarkDelay:
            pass # edge_index and edge_attr in this case do not need other assignments
        dataset.append(Data(
            x=torch.from_numpy(xs[t]), 
            y=torch.tensor(ys[t], dtype=float),
            edge_index=edge_index, 
            edge_attr=edge_attr))
    return dataset
    
    
AGENT_TYPE = {subcls.__name__.partition('Ag')[0].lower(): subcls for subcls in Agent.get_subclasses()}


class HandleDisabledMarkDelay(Exception):
    pass