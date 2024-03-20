import numpy as np

class logger(object):
    def __init__(self, keys:list):
        self.logs = {k: [] for k in keys}
        self.keys = keys

    def add_key(self, key):
        self.keys.append(key)
        self.logs[key] = []

    def log(self, key, value):
        self.logs[key].append(value)

    def get_last(self, key):
        return self.logs[key][-1]

    def reload(self):
        self.logs = {k: [] for k in self.keys}

    def get(self, key):
        return self.logs.get(key)
