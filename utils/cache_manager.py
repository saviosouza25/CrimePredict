import streamlit as st
import pandas as pd
import time
from typing import Any, Optional
import hashlib
import json

class CacheManager:
    """Enhanced cache manager with TTL and memory optimization"""
    
    @staticmethod
    def get_cache_key(func_name: str, *args, **kwargs) -> str:
        """Generate unique cache key from function name and parameters"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def is_cache_valid(cache_time: float, ttl: int) -> bool:
        """Check if cached data is still valid"""
        return time.time() - cache_time < ttl
    
    @staticmethod
    def get_cached_data(key: str, ttl: int) -> Optional[Any]:
        """Retrieve cached data if valid"""
        if key in st.session_state:
            data, cache_time = st.session_state[key]
            if CacheManager.is_cache_valid(cache_time, ttl):
                return data
            else:
                # Remove expired cache
                del st.session_state[key]
        return None
    
    @staticmethod
    def set_cached_data(key: str, data: Any) -> None:
        """Store data in cache with timestamp"""
        st.session_state[key] = (data, time.time())
    
    @staticmethod
    def clear_cache() -> None:
        """Clear all cached data"""
        keys_to_remove = [key for key in st.session_state.keys() if isinstance(st.session_state[key], tuple)]
        for key in keys_to_remove:
            del st.session_state[key]
