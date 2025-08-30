import time
from functools import wraps


def profile_time(func):
    """Decorator to profile function execution time"""

    @wraps(func)
    def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[PROFILE] {func.__name__} took {execution_time:.4f}s")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(
                f"[PROFILE] {func.__name__} failed after {execution_time:.4f}s - {str(e)}"
            )
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[PROFILE] {func.__name__} took {execution_time:.4f}s")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(
                f"[PROFILE] {func.__name__} failed after {execution_time:.4f}s - {str(e)}"
            )
            raise

    # Return appropriate wrapper based on whether function is async
    import inspect

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper