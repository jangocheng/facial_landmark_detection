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


def kaggle_data(filename, valid_percentage):
    df = pd.read_csv(filename)
    cols = df.columns[:-1]

    df = df.dropna()
    df.Image = df.Image.apply(
        lambda img: np.fromstring(img, sep=' ') / 255.0
    )

    images = np.vstack(df.Image)
    images = images.reshape((-1, 96, 96, 1))

    landmarks = df[cols].values / 96

    # extended_images = []
    # extended_landmarks = []

    # for i in range(images.shape[0]):
    #     extended_images.append(np.fliplr(images[i]))
    #     extended_landmarks.append(
    #         [1 - landmarks[i][j] if j % 2 == 0 else landmarks[i][j] for j in range(30)])

    # extended_images = np.array(extended_images)
    # extended_landmarks = np.array(extended_landmarks)

    # images = np.concatenate((images, extended_images), axis=0)
    # landmarks = np.concatenate((landmarks, extended_landmarks), axis=0)

    valid_size = int(images.shape[0] * valid_percentage)

    return {'train': DataSet(images[valid_size:], landmarks[valid_size:]), 'valid': DataSet(images[:valid_size], landmarks[:valid_size])}
