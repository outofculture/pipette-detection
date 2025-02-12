import os, threading, queue
import numpy as np
from PIL import Image


def load_training_data(path):
    if os.path.exists(os.path.join(path, 'pos.csv')):
        return TrainingData(path)
    else:
        return MultiLevelTrainingData(path)


class TrainingData:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.last_batch = None

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        self.index = []
        fh = open(os.path.join(data_path, 'pos.csv'))
        for line in fh.readlines():
            img_file, z, row, col, snr = line.split(',')[:5]
            self.index.append((os.path.join(data_path, img_file), float(z), float(row), float(col), float(snr)))
        self.image_shape = self[0][0].shape

    def __len__(self):
        return len(self.index)
    
    def __getslice__(self, sl):
        # return a new TrainingData with a slice of the index
        result = TrainingData()
        result.index = self.index[sl]
        result.image_shape = self.image_shape
        return result

    def get_arrays(self):
        """Return a tuple of (images, positions) as numpy arrays
        """
        images = []
        positions = []
        for i in range(len(self)):
            img,pos = self[i]
            images.append(img)
            positions.append(pos)
        return np.stack(images), np.stack(positions)
        
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__getslice__(item)
        img_file, z, row, col, snr = self.index[item]
        img = np.asarray(Image.open(img_file))
        pos = np.array([z, row, col, snr])
        return img, pos
    
    def generator(self, batch_size):
        preloader = Preloader(self, batch_size)
        try:
            while True:
                next_data = preloader.get_next()
                if next_data is None:
                    return
                self.last_batch = next_data
                yield next_data
        finally:
            preloader.close()

    def split(self, proportions):
        start = 0
        parts = []
        for p in proportions:
            stop = start + int(len(self) * p)
            part = TrainingData()
            part.index = self.index[start:stop]
            part.image_shape = self.image_shape
            start = stop
            parts.append(part)
        return parts

    @classmethod
    def join(self, data):
        all_data = TrainingData()
        all_data.index = []
        for d in data:
            all_data.index += d.index
        all_data.image_shape = data[0].image_shape
        return all_data


class MultiLevelTrainingData:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.levels = {}
        if data_path is not None:
            for level in sorted(os.listdir(data_path)):
                self.levels[level] = TrainingData(os.path.join(data_path, level))

    def __len__(self):
        return sum([len(self.levels[level]) for level in self.levels])

    @property
    def level_slices(self):
        slices = {}
        start = 0
        for level in self.levels:
            stop = start + len(self.levels[level])
            slices[level] = slice(start, stop)
            start = stop
        return slices

    def __getslice__(self, sl):
        # return a new MultiLevelTrainingData with slices of each level
        parts = {}
        for level in self.levels:
            parts[level] = self.levels[level][sl]
        result = MultiLevelTrainingData()
        result.levels = parts
        return result

    def __getitem__(self, item):
        # return a new MultiLevelTrainingData with slices of each level
        if isinstance(item, slice):
            return self.__getslice__(item)
        else:
            raise Exception("MultiLevelTrainingData does not support indexing (only slicing)")

    def get_arrays(self):
        # return a concatenated arrays of all levels
        images = []
        positions = []
        for level in self.levels:
            level_images, level_positions = self.levels[level].get_arrays()
            images.append(level_images)
            positions.append(level_positions)
        return np.concatenate(images), np.concatenate(positions)

    def split(self, proportions):
        # return list of MultiLevelTrainingData split from individual parts
        parts = {}
        for level in self.levels:
            parts[level] = self.levels[level].split(proportions)
        result = []
        for i in range(len(proportions)):
            part = MultiLevelTrainingData()
            part.levels = {level: parts[level][i] for level in self.levels}
            result.append(part)
        return result

    def get_level(self, level):
        return self.levels[level]

    def generator(self, batch_size):
        gens = [self.levels[level].generator(batch_size) for level in self.levels]
        while True:
            # get the next batch from each level, then concatenate and shuffle
            batches = [next(gen) for gen in gens]
            images = np.concatenate([b[0] for b in batches])
            positions = np.concatenate([b[1] for b in batches])
            indices = np.arange(len(images))
            np.random.shuffle(indices)
            for i in range(0, len(images), batch_size):
                yield images[indices[i:i+batch_size]], positions[indices[i:i+batch_size]]


class Nopealizer:
    """Does not normalize anything"""
    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x


class Normalizer:
    """Normalizes a range of values to [-1, 1]

    Parameters
    ----------
    range : list
        List of [[zmin, rowmin, colmin, ...], [zmax, rowmax, colmax, ...]]"""
    def __init__(self, range):
        range = np.array(range)
        diff = range[1] - range[0]
        self.scale = 2 / diff
        self.offset = possible_range[0] + diff / 2

    def normalize(self, x):
        return (x - self.offset) * self.scale

    def denormalize(self, x):
        return (x / self.scale) + self.offset


class ImageNormalizer:
    def normalize(self, img):
        return img / 255

    def denormalize(self, img):
        return img * 255


class OneHotNormalizer:
    """Converts a range of values to a one-hot vector

    Parameters
    ----------
    range : list
        List of [[zmin, rowmin, colmin], [xmax, rowmax, colmax]]
    size : int
        Size of the one-hot vector
    """
    def __init__(self, range, size):
        self.range = np.array(range)
        self.size = size

    def normalize(self, x):
        """
        Parameters
        ----------
        x : array-like
            2D array of [[z, row, col], ...]
        """
        x = np.array(x)
        assert x.ndim == 2 and x.shape[1] == 3

        x = np.clip(x, self.range[0], self.range[1])
        x = (x - self.range[0]) / (self.range[1] - self.range[0])
        x = np.floor(x * (self.size - 1)).astype('uint32')
        result = np.zeros((x.shape[0], 3, self.size))
        for i in (0, 1, 2):
            result[:, i, x[:, i]] = 1
        return result

    def denormalize(self, x):
        """
        Parameters
        ----------
        x : array-like
            3D array of one-hot vectors like [[[z vector], [row vector], [col vector]], ...]
        """
        x = np.array(x)
        assert x.ndim == 3 and x.shape[1] == 3 and x.shape[2] == self.size
        result = np.zeros((len(x), 3))
        for i in (0, 1, 2):
            result[:, i] = np.argmax(x[:, i], axis=-1) / self.size
        result = (result * (self.range[1] - self.range[0])) + self.range[0]
        return result


class Preloader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self.preload, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        while not self.queue.empty():
            self.queue.get()

    def preload(self):
        index = 0
        while self.running:
            stop = index + self.batch_size
            if stop > len(self.data):
                index = 0
                stop = index + self.batch_size
            chunk = self.data[index:stop].get_arrays()
            index = stop
            self.queue.put(chunk)
        self.queue.put(None)

    def get_next(self):
        # if self.queue.empty():
        #     print("Warning: preloader queue is empty (this can slow down training)")
        return self.queue.get()
