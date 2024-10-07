import functools
import hashlib
import json

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.lru = []

    def get(self, key):
        if key in self.cache:
            self.lru.remove(key)
            self.lru.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.lru.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.lru.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.lru.append(key)

cache = LRUCache(100)

def cached_llm_completion(model, messages):
    key = hashlib.md5(json.dumps((model, messages)).encode()).hexdigest()
    cached_response = cache.get(key)
    if cached_response:
        return cached_response
    
    response = litellm.completion(model=model, messages=messages, stream=True)
    cache.put(key, response)
    return response