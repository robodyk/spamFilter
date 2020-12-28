from math import floor, log


def read_classification_from_file(filepath):
    dictionary = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            classification = line.split()
            try:
                dictionary[classification[0]] = classification[1]
            except IndexError:
                pass
    return dictionary

def order_of_magnitude(num):
    return floor(log(num, 10))
