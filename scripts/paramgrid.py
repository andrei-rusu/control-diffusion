import argparse
import json
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

GRID = [
    {'m:layer_name': ['GAT', 'GIN'],
     'a:typ': ['sl'],
     'a:rl_sampler': ['', 'softmax'],
     'control_initial_known': [0.25, 0.5],
     'control_after': [3, 5]
    },
]
sampler = None

def create_sampler(sample=0, state=1):
    global sampler 
    sampler = list(ParameterSampler(GRID, n_iter=sample, random_state=state)) if sample else ParameterGrid(GRID)
    return sampler

def len_sampler():
    return len(sampler)
    
def get_job_ids():
    print(f'0-{len(sampler) - 1}')

def get_parameters_for_id(job_id, agent_file=0):
    # the order of variables is given by string name in decreasing order
    params = sampler[job_id]
    if agent_file:
        epidemic_params = []
        with open('agent_config.json', 'r', encoding='utf8') as handle:
            agent = json.loads(handle.read())     
        with open('amodel_config.json', 'r', encoding='utf8') as handle:
            amodel = json.loads(handle.read())
        for k, v in params.items():
            if k.__contains__('a:'):
                agent[k[2:]] = v
            elif k.__contains__('m:'):
                amodel[k[2:]] = v
            else:
                epidemic_params.append(v)
        with open(f'temp/agent_{job_id}.json', 'w', encoding='utf8') as handle:
            json.dump(agent, handle)     
        with open(f'temp/amodel_{job_id}.json', 'w', encoding='utf8') as handle:
            json.dump(amodel, handle)
    else:
        epidemic_params = params.values()
    std_write(*epidemic_params)
    return len(epidemic_params)
    
def std_write(*args, **kwargs):
    for arg in args:
        print(round(arg, 2) if isinstance(arg, float) else arg, end=' ', **kwargs)
        
def modify_json(job_id, agent_params, amodel_params):
    with open('agent_config.json', 'r', encoding='utf8') as handle:
        agent = json.loads(handle.read())     
    with open('amodel_config.json', 'r', encoding='utf8') as handle:
        amodel = json.loads(handle.read())
    params = sampler[job_id]
    agent.update({k[2:]:v for k,v in params.items() if k.__contains__('a:')})
    amodel.update({k[2:]:v for k,v in params.items() if k.__contains__('m:')})
    with open(f'temp/agent_{job_id}.json', 'r', encoding='utf8') as handle:
        json.dump(agent, handle)     
    with open(f'temp/amodel_{job_id}.json', 'r', encoding='utf8') as handle:
        json.dump(amodel, handle)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--sample', type=int, default=0)
    argparser.add_argument('--state', type=int, default=0)
    argparser.add_argument('--id', type=int, default=0)
    argparser.add_argument('--agent_file', type=int, default=1)
    # parse_known_args is needed for the script to be runnable from Notebooks
    args, _ = argparser.parse_known_args()
    create_sampler(args.sample, args.state if args.state else args.sample)
    get_parameters_for_id(args.id, args.agent_file)