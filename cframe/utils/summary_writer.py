import os
import pandas as pd
import json

class SummaryWriter(object):
    def __init__(self, *keys):
        self.logs = dict()
        for key in keys:
            self.logs[key] = []

    def append(self, **items):
        for k, v in items.items():
            self.logs[k].append(v)

    def reset(self):
        for k, v in self.logs.items():
            self.logs[k] = []

    def save(self, path=None):
        save_path = os.path.join(path,  'train_log.json')
        with open(save_path, 'w') as f:
            json.dump(self.logs, f)

    def get(self, name):
        return self.logs[name]


if __name__ == '__main__':
    writer = SummaryWriter(*['train_loss', 'valid_loss'])
    writer.append(**dict(
        train_loss=91,
        valid_loss=100
    ))
    writer.save(path='/home/he')
