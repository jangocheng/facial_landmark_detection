import numpy as np
import pandas as pd


class DataSet(object):
    def __init__(self, images, targets, reshape=False):
        if reshape:
            images = images.reshape(
                images.shape[0], images.shape[1] * images.shape[2])
        self._images = images
        self._targets = targets
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True):
        '''Return the next batch_size example from this data set.'''
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._targets = self.targets[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            targets_rest_part = self._targets[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._targets = self.targets[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            targets_new_part = self._targets[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((targets_rest_part, targets_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._targets[start:end]


def kaggle_data(filename, test=False):
    df = pd.read_csv(filename)
    cols = df.columns[:-1]

    df = df.dropna()
    df.Image = df.Image.apply(
        lambda img: np.fromstring(img, sep=' ') / 255.0
    )

    images = np.vstack(df.Image)
    images = images.reshape((-1, 96, 96, 1))

    if test:
        landmarks = None
    else:
        landmarks = df[cols].values / 96

    return DataSet(images, landmarks)
