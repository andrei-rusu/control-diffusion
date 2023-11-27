# EpiCURB: Epidemic Control Using Reinforcement-learning for Budget allocation

This is a repository that contains agent implementations aimed at controlling diffusion processes over networks.
The primary use case for these agents is to limit the spread of epidemics through *targeted testing, contact tracing and vaccination*.
For more details, please refer to our EAI PervasiveComputing Conference paper: <a href="https://link.springer.com/chapter/10.1007/978-3-031-34586-9_14" target="_blank">Rusu et al., 2022</a>.

To use, please install the package via `pip`, alongside the Epidemic Simulator linked <a href="https://github.com/andrei-rusu/contact-tracing-model" target="_blank">here</a>.

Note, other simulators could also be used together with this package. To do so, one needs to do the following:
* The simulator needs to instantiate the Agent object through the factory method `from_dict`:

    ```python
    from control_diffusion import Agent
    agent = Agent.from_dict(**agent_params)
    ```
* The simulator has to call the agent's `control` method at each timestamp with the appropriate parameters.

    ```python
    agent.control(network, control_day, initial_known_ids, net_changed, missed_days)
    ```
* Optionally, the simulator can call the agent's `finish` method just before the full simulation is terminated, for logging purposes and offline learning.
    ```python
    agent.finish(total_infected, args_log)
    ```

The package can also be used to create highly customizable graph neural network models by calling the factory method ``Agent.model_from_dict``. Additionally, we provide implementations for various helper objects and routines in the ``control_diffusion.general_utils`` module, such as replay buffers, list wrappers compatible with ``torch_geometric``, printing logic enhancements for ``tqdm`` and others.