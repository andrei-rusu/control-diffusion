import tqdm
import inspect
from sys import stdout
from contextlib import contextmanager

def tqdm_print(*args, sep=' ', **kwargs):
    to_print = sep.join((str(arg) for arg in args))
    tqdm.tqdm.write(to_print, **kwargs)
        
@contextmanager
def redirect_to_tqdm():
    """Context manager to allow tqdm.write to replace the print function"""
            
    # Store builtin print
    old_print = print
    try:
        # Globaly replace print with tqdm.write
        inspect.builtins.print = tqdm_print
        yield
    finally:
        inspect.builtins.print = old_print
        
def tqdm_redirect(*args, **kwargs):
    with redirect_to_tqdm():
        for x in tqdm.tqdm(*args, file=stdout, **kwargs):
            yield x
            
def save_model(model, file='rl', path='saved/ckp'):
    import torch
    filename = f'{path}/{file}.pt'
    torch.save(model.state_dict(), filename)
    
def load_model(model, file='rl', path='saved/ckp'):
    import torch
    filename = f'{path}/{file}.pt'
    model.load_state_dict(torch.load(filename))
    return model