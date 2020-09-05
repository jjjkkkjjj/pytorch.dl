import numpy as np

def decision(p=0.5):
    val = np.random.choice(2, 1, p=[1-p, p])[0]
    return val == 1

class Compose(object):
    def __init__(self, augmentaions):
        self.augmentaions = augmentaions

    def __call__(self, img, *targets):
        for t in self.augmentaions:
            img, targets = t(img, *targets)
        return img, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.augmentaions:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string