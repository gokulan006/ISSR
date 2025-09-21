"""
Timeout handling utilities for MIE Classifier
Prevents hanging during Ollama calls and other long operations
Cross-platform compatible (Windows, macOS, Linux)
"""

import signal
import time
import threading
import platform
from contextlib import contextmanager
from typing import Any, Callable, Optional

class TimeoutError(Exception):
    """Raised when operation times out"""
    pass

@contextmanager
def timeout_handler(seconds: int):
    """
    Context manager for timeout handling (cross-platform)
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    if platform.system() == 'Windows':
        # Windows doesn't support SIGALRM, use threading instead
        timeout_occurred = threading.Event()
        
        def timeout_worker():
            time.sleep(seconds)
            timeout_occurred.set()
        
        timeout_thread = threading.Thread(target=timeout_worker)
        timeout_thread.daemon = True
        timeout_thread.start()
        
        try:
            yield
            if timeout_occurred.is_set():
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
        finally:
            timeout_occurred.set()  # Stop the timeout thread
    else:
        # Unix-like systems (macOS, Linux)
        def timeout_signal_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore old handler and cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)

def safe_request_with_timeout(func: Callable, timeout: int = 30, *args, **kwargs) -> Any:
    """
    Safely execute a function with timeout
    
    Args:
        func: Function to execute
        timeout: Timeout in seconds
        *args, **kwargs: Arguments for the function
        
    Returns:
        Function result or None if timeout
        
    Raises:
        TimeoutError: If operation times out
    """
    try:
        with timeout_handler(timeout):
            return func(*args, **kwargs)
    except TimeoutError:
        print(f"⚠️  Operation timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"⚠️  Error during operation: {e}")
        return None

def progress_indicator(message: str, duration: int = 3):
    """
    Show a simple progress indicator
    
    Args:
        message: Message to display
        duration: Duration in seconds
    """
    print(f"⏳ {message}...", end="", flush=True)
    time.sleep(0.5)
    for i in range(duration):
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" ✅")
