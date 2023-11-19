import torch


class ActionSampler:
    """
    A class for sampling actions from a distribution over candidate nodes.

    Attributes:
        rl_sampler (str): The type of RL sampler to use. Can be one of 'greedy', 'softmax', 'log', 'nvidia', 'escort'.
        dist_over_sample (int): The type of distribution to use over the candidate nodes. Can be one of 0, 1, 2. This controls whether the probability distribution 
            is over all nodes (0), candidate nodes only (1), or all nodes but with out-of-candidates set to very low scores (2).
        sample_from_dist (bool): Whether to sample actions from the `pi.Categorical` distribution over all nodes (True) or from the `indices_sampler` (False).
            The latter allows sampling from either all nodes or a subset of candidate nodes efficiently.
        logp_from_score (bool): Whether to calculate the log-probabilities of the actions from the raw scores or from the `pi.Categorical` distribution.
        force_cndt_sample (bool): Only used if `sample_from_dist` is True, this controls whether to force the sampled actions to be from the input candidate nodes.
            This can result in large computational overhead, and may even result in an infinite loop if the input candidate nodes have very low probabilities.
        episode (int): The current episode number.
        eps (float): The episod-specific epsilon noise value for the action sampler.
        control_iter (int): The current control iteration number.
        rng (numpy.random.RandomState): The random number generator created for the Agent.
        n_nodes (int): The number of nodes in the Network to be controlled.
    """
    def __init__(self, agent):
        """
        Initializes the ActionSampler class.

        Args:
            agent: An instance of the agent class.
        """
        self.rl_sampler = agent.rl_sampler
        self.dist_over_sample = agent.dist_over_sample
        self.logp_from_score = agent.logp_from_score
        self.sample_from_dist = agent.sample_from_dist
        self.force_cndt_sample = agent.force_cndt_sample
        self.episode = agent.episode
        self.eps = agent.eps
        self.control_iter = agent.control_iter
        self.rng = agent.rng
        self.n_nodes = agent.n_nodes
        
    def score_to_prob(self, scores, size=10):
        """
        Converts a tensor of scores into a probability distribution over actions per node.

        Args:
            scores (torch.Tensor): A tensor of shape (batch_size, n_nodes) containing the scores for each node.
            size (int): The number of actions to sample from the distribution.

        Returns:
            A tuple containing:
            - pi (torch.distributions.Categorical): A Categorical distribution built using action probabilities. 
                This is primarily used for calculating the log-probabilities of the actions.
            - indices_sampler (callable): A function that can be used to sample action indices according to the action raw probabilities.
        """
        if self.rl_sampler == 'greedy':
            # partial sorting of node scores and taking maximum
            greedy = scores.topk(size, dim=-1, sorted=False)
            mask = fill_with_value(scores, greedy, value=False, rest=True)
            # make logp as log(eps) for non-greedy actions, and logp + log(eps) for greedy actions
            scores = scores.masked_fill(mask, 0) + torch.log(self.eps)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            # random choice or greedy action
            indices_sampler = lambda: self.rng.choice(scores.shape[-1], size=size, replace=False) \
                                if self.rng.random() < self.eps else greedy
        else:
            if self.rl_sampler == 'softmax':
                probs = torch.nn.functional.softmax(scores / self.eps, dim=-1)
            elif self.rl_sampler == 'log':
                probs = torch.exp(torch.nn.functional.log_softmax(scores / self.eps, dim=-1))
            elif self.rl_sampler == 'nvidia':
                probs = torch.clamp(scores - scores.min(dim=-1, keepdim=True).values + self.eps, min=0)
            elif self.rl_sampler == 'escort':
                probs = scores.abs() ** (1 / self.eps)
            indices_sampler = lambda: torch.multinomial(probs, num_samples=size)

        pi = torch.distributions.Categorical(probs=probs)
        return pi, indices_sampler
    
    def get_logp_and_entropy(self, scores, actions, nodes=None, sum_logp=True):
        """
        Computes the log-probabilities and entropy of the given actions, given the scores of the candidate nodes.

        Args:
            scores (torch.Tensor): A tensor of shape (batch_size, n_nodes) containing the scores of the candidate nodes.
            actions (torch.Tensor): A tensor of shape (batch_size, n_actions) containing the indices of the selected actions.
            nodes (Optional[torch.Tensor]): A tensor of shape (batch_size, n_nodes) containing the indices of the candidate nodes. Default is None. 
                If not None (e.g. when `zero_outsample_logits` is True), nodes that are NOT in this list will be given a very low score (approx 0 probability of selection).
            sum_logp (bool): Whether to return the sum of log-probabilities or the log-probabilities themselves.

        Returns:
            tuple - Tuple containing the log-probabilities and entropy of the given actions. If `sum_logp` is True, the log-probabilities are summed along the last dimension,
                and the entropy is returned as the negative sum of log-probabilities. Otherwise, the log-probabilities and entropy are returned separately as tensors of 
                shape (batch_size,).
        """
        scores = scores.view(-1, self.n_nodes)
        actions = actions.view(scores.shape[0], -1)
        device = scores.device
        # fill the scores with a value equivalent to prob=0 for the vertices that are NOT in the list of candidate 'nodes' iff nodes is not None
        # nodes are None, when agent.zero_outsample_logits is False, or when the agent is not using a candidate set
        if nodes is not None:
            mask = fill_with_value(scores, nodes.view(scores.shape[0], -1).to(device), value=False, rest=True)
            scores = scores.masked_fill(mask, -torch.abs(10 * scores.min()))
            
        pi, _ = self.score_to_prob(scores, size=actions.shape[-1])
        
        # we can choose to extract the true logp from the pi.Categorical distribution over all nodes, carrying out sampling without replacement corrections 
        # over unnormalized logits; or compute a rough estimate of these logp, without any corrections, directly from some representation of the logits.
        if self.sample_from_dist:
            logp = pi.logits.gather(1, actions)
            # proposed by Meirom et al for stabilization of softmax, but unclear whether this improves performance
            stabilize = self.rl_sampler in ('softmax', 'log')
            if stabilize:
                logp = torch.max(logp, torch.log(torch.tensor([self.eps], device=device)))
            
            # if more than one action, compute corrections to logp to reflect sampling without replacement
            # this is reflected by conditionals log P(action | previous actions) = log pi.probs[action] - log (sum of pi.probs[remaining actions])
            if actions.shape[1] >= 2:
                # start with computing the correction term for the last action, which will be log(sum of all probs but the last)
                mask = fill_with_value(scores, actions[:, :-1], value=False, rest=True)
                partial_prob = pi.probs[mask].view(pi.probs.shape[0], -1)
                if stabilize:
                    partial_prob += self.eps
                partial_prob_sum = partial_prob.sum(dim=1)
                correction_term = torch.log(partial_prob_sum)
                # if more than two actions, further add to the correction term to obtain the corresponding conditionals
                # note for the last action, this was computed above, while the first action is not conditioned on any previous actions
                if actions.shape[1] > 2:
                    selected_nodes_prob = torch.gather(pi.probs, 1, actions[:, 1:-1:].flip(dims=[1]))
                    cumsum_prob = torch.cumsum(selected_nodes_prob, axis=1) + partial_prob_sum.unsqueeze(dim=1)
                    correction_term += torch.log(cumsum_prob).sum(axis=1)
            # if single actions per timestamp, no correction needed
            else:
                correction_term = 0
            # correct logp to reflect sampling without replacement
            logp_sum = logp.sum(dim=-1) - correction_term
        else:
            # `logp_from_score` controls whether we consider the raw scores to be logits, or we utilize the pi.Categorical log probabilities
            # if the first, the transformations carried in `score_to_prob` are NOT considered for the logp calculation
            logp = scores.gather(1, actions) if self.logp_from_score else pi.log_prob(actions)
            logp_sum = logp.sum(dim=-1)
            
        return (logp_sum, -logp_sum) if sum_logp else (logp, pi.entropy())
    
    def get_action(self, scores, nodes, size=10, sum_logp=False):
        """
        Samples actions from a probability distribution over nodes, given a set of scores.

        Args:
            scores (torch.Tensor): A tensor of shape (batch_size, n_nodes) containing the scores for each node.
            nodes (list): A list of candidate nodes to sample from.
            size (int, optional): The number of actions to sample. Defaults to 10.
            sum_logp (bool, optional): Whether to return the sum of the log probabilities of the sampled actions. Defaults to False.

        Returns:
            tuple - Tuple that contains:
                - actions (torch.Tensor): A tensor of shape (batch_size, size) containing the sampled actions.
                - logp (torch.Tensor): A tensor of shape (batch_size,) containing the log probabilities of the sampled actions.
                - entropy (torch.Tensor): A tensor of shape (batch_size,) containing the entropy of the probability distribution.
        """
        scores = scores.view(-1, self.n_nodes)
        batch_dim = scores.shape[0]
        device = scores.device

        if not self.sample_from_dist and self.dist_over_sample == 1:
            scores = scores[:, nodes]
        # dist_over_sample = 2 is a special case, where the dist is over all nodes, but out-of-candidate nodes are given a very low score
        elif self.dist_over_sample == 2:
            mask = fill_with_value(scores, nodes.view(batch_dim, -1), value=False, rest=True)
            scores = scores.masked_fill(mask, -torch.abs(10 * scores.min()))
        max_size = min(size, scores.shape[-1])
        
        # we can choose to conduct a true sampling without replacement from a pi.Categorical distribution over all nodes, trialing until we get the desired number of actions
        # or utilize indices_sampler to sample node actions from the raw logits + torch.multinomial, utilizing pi.Categorical only for logp and/or entropy calculation
        if self.sample_from_dist:
            actions = torch.full(size=(len(scores), size), fill_value=-1, device=device, dtype=torch.long)
            if self.force_cndt_sample:
                nodes_set = set(nodes.tolist())
            logp = []
            entropy = 0
            # sample without replacement from the pi.Categorical distribution over all nodes
            for k in range(max_size):
                # scores need to be rescaled after actions were picked
                scores_after_sample = scores if k == 0 else fill_with_value(scores, actions[:, :k], value=0)
                pi, _ = self.score_to_prob(scores_after_sample, size=1)
                trial_sample = pi.sample()
                for i, node in enumerate(trial_sample):
                    while node in actions[i] or (self.force_cndt_sample and node.item() not in nodes_set):
                        node = pi.sample()[i]
                    trial_sample[i] = node
                actions[:, k] = trial_sample
                logp.append(pi.log_prob(trial_sample))
                entropy += pi.entropy()
            logp = torch.column_stack(logp)
        else:
            # get the pi.Categorical probability distribution for logp and entropy calcuation, but allow sampling using the raw logits + torch.multinomial
            pi, indices_sampler = self.score_to_prob(scores, size=max_size)
            actions = indices_sampler()
            # logp_from_score controls whether we consider the raw scores to be logits, or we utilize the pi.Categorical log probabilities
            # if the first, the transformations carried in score_to_prob are NOT considered in the logp calculation
            logp = scores.gather(1, actions) if self.logp_from_score else pi.log_prob(actions)
            entropy = pi.entropy()
            
        # if dist_over_sample = 1, we need to convert the sampled actions (indices inside candidate node list) to the corresponding true node indices    
        if self.dist_over_sample == 1:
            actions = nodes[actions.ravel()].reshape(actions.shape)
        return actions, logp.sum(dim=-1) if sum_logp else logp, entropy
    
    
def fill_with_value(scores, selected_nodes, value=0, rest=None):
    """
    Fills the given scores tensor with the specified `value` at the indices of the selected nodes, and with `rest` at the remaining indices.
    This is NOT an in-place operation.

    Args:
        scores (torch.Tensor): The tensor to fill with values.
        selected_nodes (torch.Tensor): The indices of the nodes to fill with the specified value.
        value (float or int, optional): The value to fill the tensor with. Defaults to 0.
        rest (float or int, optional): The value to fill the remaining tensor with. Defaults to None.

    Returns:
        torch.Tensor: The filled tensor.
    """
    if rest is not None:
        scores = torch.full(scores.shape, fill_value=rest, device=scores.device, dtype=type(rest))
    scores = scores.scatter(1, selected_nodes, value)
    return scores