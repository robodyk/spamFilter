import os
from mail import Mail

class Corpus:
    def __init__(self, file_path):
        self.path = file_path

    def emails(self):
        for filename in os.listdir(self.path):
            if filename[0] == '!':
                continue
            with open(self.path + '/' + filename, 'r', encoding='utf-8') as f:
                yield filename, Mail(f.read(), filename)
