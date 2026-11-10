"""The input transform the ball lifting model expects at inference (training augmentations were removed)."""

import numpy as np

from ball_uplifting.uplifting_multisequence.helper import HEIGHT, WIDTH


class Compose:
    '''
    Composes several transforms together.
    '''

    def __init__(self, transforms):
        '''
        transforms (list of objects) – list of transforms to compose.
        '''
        self.transforms = transforms

    def __call__(self, data):
        '''
        data (dict) – Dictionary with keys 'r_img' and 'table_img'.
        '''
        for t in self.transforms:
            data = t(data)
        return data


class NormalizeImgCoords:
    '''
    Normalize image coordinates to the range [0, 1] using HEIGHT and WIDTH. Apply transformation after RandomizeDetections!!!
    '''

    def __init__(self):
        pass

    def __call__(self, data):
        r_img = data['r_img']
        table_img = data['table_img']
        r_img[..., :2] = r_img[..., :2] / np.array([WIDTH, HEIGHT])
        table_img[..., :2] = table_img[..., :2] / np.array([WIDTH, HEIGHT])
        data['r_img'] = r_img
        data['table_img'] = table_img
        return data


def get_transforms():
    '''The test-time transforms: only the image-coordinate normalization.'''
    return Compose([NormalizeImgCoords()])
