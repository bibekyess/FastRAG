from functools import wraps
from time import perf_counter
import logging
from contextlib import contextmanager
from typing import Callable


def get_execution_time(func, level="INFO"):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = await func(*args, **kwargs)
        end_time = perf_counter()
        duration = end_time - start_time
        log_fcn = getattr(logging, level.lower())
        
        func_name = func.__name__        
        log_fcn(f"Function {func_name} executed in {duration:.4f} seconds")
        return result
    return wrapper
        

@contextmanager
def log_time_context(desc: str = "Section", level="DEBUG") -> Callable[[], float]:
    """
    with log_time_context("func_1") as elapsed_time_1:
        func_1()
    
    time_func_1 = elapsed_time_!
    """
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1 # Callable returning the elapsed time as float
    t2 = perf_counter()
    duration = t2 - t1
    log_fcn = getattr(logging, level.lower())
    log_fcn(f"Section {desc} executed in {duration:.4f} seconds")
    
