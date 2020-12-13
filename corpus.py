import os


class Corpus:
    def __init__(self, file_path):
        self.path = file_path

    def emails(self):
        for filename in os.listdir(self.path):
            if not filename[0] == 0:
                continue
            with open(self.path + '/' + filename, 'r', encoding='utf-8') as f:
                yield filename, f.read()
