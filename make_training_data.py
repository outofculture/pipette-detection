"""
Martin notes, 26 June:

    This will need to eventually handle different pitches, pull-shapes, magnifications and lighting conditions. Maybe
    also yaws, but seeing as we know the expected yaw of any pipette in ACQ4, we could just rotate the live image to
    match our model's training data, et voilà! If this eventually gets a reasonably accurate detector, then teaching
    it to detect at all yaws would be useful to allow multiple pipettes to be identified simultaneously.

    "Difficulty" is not objectively that. While adding noise will make images harder to parse, other factors (such as
    z distance, tip out of FoV, or just total pixels impacted by pipette) also play a role. The combination of all
    these factors could possibly be used to quantify how accurate a pipette-detection could be. We could maybe help
    this by building toward using a detection mask + semantic segmentation, rather than just tip position.

"""
import argparse
import os
import threading
from queue import Queue
from typing import Tuple

import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import MetaArray
import yaml


class PipetteTemplate:
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.file = npz_file
        self.image = data['image_data'] / data['image_data'].max()
        # Set background to 0 (right side of template image should be all background)
        self.image -= self.image[:, :, -10:].mean(axis=1).mean(axis=1)[:, None, None]
        self.pos = data['pipette_pos']  # frame, row, col
        self.z = data['z_um']
        self.pixel_size = data['pixel_size']
        self.shape = self.image.shape

    def get_image(self, z=0, flip=True):
        """Return template image with the closest possible Z value (where z is
        the height of the pipette relative to the focal plane), along with 3D
        pipette position (z_um, row, col), and pixel size.

        If flip is True, randomly flip vertically around the pipette tip

        If z is None, choose randomly
        """
        if z is None:
            ind = np.random.randint(self.image.shape[0])
        else:
            ind = np.argmin(np.abs(self.z - z))
        img = self.image[ind]
        pos = np.array((self.z[ind],) + tuple(self.pos[1:]))

        # randomly flip vertically around the pipette tip
        if flip and np.random.random() > 0.5:
            img = img[::-1]
            pos[1] = img.shape[0] - pos[1]

        return img, pos, self.pixel_size

    def add_to_image(self, z, dst_arr, pip_pos, dst_pixel_size, amp=1, angle=0, scale=1):
        """Add pipette template z to *dst_arr* such that the tip is at *pip_pos* (row, col), 
        ignoring non-overlapping areas.
        
        The template is scaled to match the pixel size of the destination image, plus an extra
        scaling factor *scale*.

        Return chosen Z position and sum of squared differences added by the template.
        """
        # get template image and pipette position
        template_arr, (template_z_um, template_row, template_col), template_px_size = self.get_image(z)
        template_pip_pos = np.array([template_row, template_col])

        px_scale = template_px_size / dst_pixel_size
        scale = scale * px_scale
        
        if angle != 0:
            # scale template image
            center1 = np.array(template_arr.shape) / 2
            template_arr = scipy.ndimage.zoom(template_arr.astype(float), scale)
            center2 = np.array(template_arr.shape) / 2

            # scale offset position (rows, cols) around the center of the image
            template_pip_pos -= center1
            template_pip_pos *= scale
            template_pip_pos += center2

            # rotate template image
            center1 = np.array(template_arr.shape) / 2
            template_arr = scipy.ndimage.rotate(template_arr, angle, reshape=False)
            center2 = np.array(template_arr.shape) / 2

            # rotate offset position (rows, cols) around the center of the image
            template_pip_pos -= center1
            rotation_matrix = [
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ]
            template_pip_pos = np.dot(rotation_matrix, template_pip_pos)
            template_pip_pos += center2

        offset = (np.array(pip_pos) - template_pip_pos).astype(int)

        # add template to destination array
        dst_rgn = np.array([offset, np.array(offset) + template_arr.shape])
        dst_rgn = np.clip(dst_rgn, 0, dst_arr.shape)
        if np.all(dst_rgn[0] < dst_rgn[1]):
            src_rgn = dst_rgn - offset
            src_subrgn = amp * template_arr[src_rgn[0,0]:src_rgn[1,0], src_rgn[0,1]:src_rgn[1,1]]
            dest_subrgn = dst_arr[dst_rgn[0,0]:dst_rgn[1,0], dst_rgn[0,1]:dst_rgn[1,1]]
            bg_subrgn = dest_subrgn.copy()
            dest_subrgn += src_subrgn

            # calculate pipette visibility
            fg = src_subrgn
            bg = bg_subrgn
            abs_fg = np.abs(fg)
            template_max = np.abs(template_arr).max() * amp
            mask = abs_fg > template_max * 0.3
            if mask.sum() == 0:
                signal = 0
                noise = 1
            else:
                masked_bg = bg[mask]
                signal = abs_fg[mask].sum()
                noise = np.abs(masked_bg - masked_bg.mean()).sum()
            snr = 0.01 * signal / noise

        else:
            template_max = signal = noise = snr = 0


        return template_z_um, {
            'template_max': template_max,
            'signal': signal,
            'noise': noise,
            'snr': snr,
        }


class PipetteTemplates:
    """Loads multiple PipetteTemplate files and allows sampling from them
    """
    def __init__(self, npz_files):
        self.templates = [PipetteTemplate(f) for f in npz_files]

    def get_image(self, z=None):
        """Select a random template and return image with the closest possible
        Z value, along with 3D pipette position (z_um, row, col)

        If z is None, choose randomly
        """
        template = np.random.choice(self.templates)
        return template.get_image(z)

    def add_to_image(self, z, dst_arr, pip_pos, dst_pixel_size, amp=1, angle=0, scale=1):
        """Add pipette template z to *dst_arr* such that the tip is at *pip_pos* (row, col),
        ignoring non-overlapping areas.

        Return chosen Z position.
        """
        i = np.random.randint(0, len(self.templates))
        template = self.templates[i]
        return template.add_to_image(z, dst_arr, pip_pos, dst_pixel_size, amp, angle=angle, scale=scale)


class NoiseData:
    """List of .ma files containing background noise to mix into training data
    """
    def __init__(self, files):
        self.files = files
        assert len(self.files) > 0, f'No noise files specified'
        self.data = None
        self.meta = None
        self.lock = threading.Lock()

    def _load_noise(self):
        with self.lock:
            if self.data is None:
                self.data = []
                self.meta = []
                for nf in self.files:
                    path, filename = os.path.split(nf)
                    meta = yaml.safe_load(open(path + '/.index', 'r'))[filename]
                    self.data.append(MetaArray.MetaArray(file=nf).asarray())
                    self.meta.append(meta)
        return self.data, self.meta

    def get_noise(self, size: int, noise_amp: float):
        all_noise, all_meta = self._load_noise()
        # select a random noise file
        i = np.random.randint(0, len(all_noise))
        noise = all_noise[i]
        px_size = all_meta[i]['pixelSize'][0]

        # select a random plane
        i = np.random.randint(0, noise.shape[0])
        noise = noise[i]
        # select a random chunk
        i = np.random.randint(0, max(1, noise.shape[0] - size))
        j = np.random.randint(0, max(1, noise.shape[1] - size))
        noise = noise[i:i+size, j:j+size].copy()
        # randomly flip / rotate
        if np.random.random() > 0.5:
            noise = noise[::-1]
        noise = np.rot90(noise, np.random.randint(0, 3))
        # normalize
        noise -= noise.min()
        noise = noise * (noise_amp / noise.max())

        return noise, px_size


class NoiseGenerator(NoiseData):
    """Generate random noise"""
    def __init__(self, noise_radii, noise_amplitudes, sin_shift=0.1, noise_exponent=2):
        self.noise_radii = noise_radii
        self.noise_amplitudes = noise_amplitudes
        self.sin_shift = sin_shift
        self.noise_exponent = noise_exponent

    def get_noise(self, size: int, noise_amp: float):
        # structured noise to look like cells / neuropil
        str_noise_len = 3
        image = NoiseGenerator.make_structured_noise(
            shape=(size, size),
            edge=np.random.normal(size=2, scale=3),
            edge_frac=np.random.normal(scale=0.2, loc=1),
            noise_radii=np.random.uniform(1, 50, size=str_noise_len),
            noise_amplitudes=10**np.random.normal(size=str_noise_len, loc=noise_amp, scale=0.2),
            sin_shift=np.random.uniform(0.05, 0.3),
            noise_exponent=2,
        )
        # unstructured noise at various scales
        image += NoiseGenerator.make_noise(
            shape=(size, size),
            amplitudes=10**np.random.normal(size=3, loc=noise_amp, scale=0.2),
            radii=[
                np.random.normal(loc=100, scale=30),
                np.random.normal(loc=10, scale=3),
                np.random.normal(loc=2, scale=1),
            ],
        )
        return image

    @staticmethod
    def make_noise(amplitudes, radii, shape):
        """Return a gaussian-smoothed noise image.
        """
        shape = np.array(shape)
        total = np.zeros(shape)
        for amplitude, radius in zip(amplitudes, radii):
            if radius > 10:
                # large radius gaussian smoothing is slow, so speed up by smoothing a smaller image, then zooming
                scale = radius / 2
                radius = 2
            else:
                scale = 1
            # generate noise
            n = np.random.normal(size=(shape//scale).astype(int))
            # gaussian smoothing
            if radius != 0:
                n = scipy.ndimage.gaussian_filter(n, (radius, radius))
            # normalize
            n *= amplitude / n.max()
            # scale up if needed
            if scale != 1:
                z = shape / n.shape
                n = scipy.ndimage.zoom(n, z)
            total += n
        return total

    @staticmethod
    def make_structured_noise(shape, edge, edge_frac, noise_radii, noise_amplitudes, sin_shift=0.1, noise_exponent=2):
        shape = np.array(shape, dtype=int)
        edge = np.array(edge, dtype=int)
        noise = NoiseGenerator.make_noise(noise_amplitudes, noise_radii, shape+np.abs(edge))
        noise = np.sin(1 / (sin_shift + noise**noise_exponent))

        starta = np.clip(edge, 0, np.inf).astype(int)
        startb = np.clip(-edge, 0, np.inf).astype(int)
        a = noise[starta[0]:starta[0]+shape[0], starta[1]:starta[1]+shape[1]]
        b = noise[startb[0]:startb[0]+shape[0], startb[1]:startb[1]+shape[1]]
        noise = a - edge_frac*b

        return noise


def scale_image(image, shape, **kwds):
    """Scale image to an exact shape (zoom does not always generate the expected shape)
    """
    if image.shape == shape:
        return image, np.array([1, 1])
    
    scale = (np.array(shape) + 1) / np.array(image.shape)
    resized = scipy.ndimage.zoom(image, scale, **kwds)
    actual_scale = np.array(resized.shape) / np.array(shape)
    return resized[:shape[0], :shape[1]], actual_scale


def make_training_data(size:int, template:PipetteTemplate, noise_data:NoiseData, difficulty:float,
                       angle_deg_stdev:float=2, pip_scale_exponent_stdev:float=0.2, source_size:int|None=None) -> Tuple[np.ndarray, Tuple[float, float, float], dict]:
    """Make a single training image with a pipette at a random position and focus depth

    Parameters
    ----------
    size : int
        Final size (width and height) of the image
    template : PipetteTemplate
        Pipette template to use
    noise_data : NoiseData
        Noise data to mix in
    difficulty : float
        Difficulty (0-1) controls signal/noise ratio, pipette focus and positioning
    angle_deg_stdev : float
        Standard deviation of random angle in degrees to rotate the pipette
    pip_scale_exponent_stdev : float
        Standard deviation of random base-10 exponent to scale the pipette
    img_scale_exponent_stdev : float
        Standard deviation of random base-10 exponent to scale the final image
    source_size : int | tuple | None
        Size of the source image (width and height) to generate (before scaling to final size).
        Default is *size*
    """
    if source_size is None:
        source_size = size
    final_shape = (size, size)
    source_shape = (source_size, source_size)
    radius = source_size * (0.1 + difficulty * 0.3)
    center = np.array(source_shape) // 2

    pip_pos = (center + np.random.uniform(-radius, radius, size=2)).astype(int)

    # scale noise amplitude such that smaller difficulty values primarily
    # differ in z range rather than noise
    # noise_amp = np.clip((difficulty - 0.2) * 50, 0, np.inf)
    noise_amp = 1 + difficulty**2 * 50

    # generate or load noise
    image, bg_px_size = noise_data.get_noise(int(source_size), noise_amp)

    image = base_image.copy()
    # add in pipette template
    z_difficulty = difficulty**0.5
    z_range = 40 * z_difficulty  # μm
    z_target = np.random.uniform(-z_range, z_range)
    z_um, stats = template.add_to_image(
        z=z_target,
        dst_arr=image,
        pip_pos=pip_pos,
        dst_pixel_size=bg_px_size,
        amp=10**np.random.normal(loc=0.2, scale=0.1),
        angle=np.random.normal(scale=angle_deg_stdev),
        scale=10**np.random.normal(scale=pip_scale_exponent_stdev),
    )

    # scale image
    image, scale = scale_image(image, final_shape, mode='nearest', order=2)
    pip_pos = pip_pos * scale

    percent_diff = np.sum(np.abs(base_image - image)) / (shape[0] * shape[1])
    if percent_diff < 0.02:  # too imperceptible; try again
        return make_training_data(shape, template, difficulty)

    # normalize image
    image = image - image.min()
    image /= image.max()

    return image, (z_um, pip_pos[0], pip_pos[1]), stats


def save_training_data(path, img_count, image, pip_pos, image_stats):
    if not os.path.exists(path):
        os.makedirs(path)
    image = Image.fromarray(image*255).convert('RGB')
    img_file = f'{img_count:05d}.jpg'
    image.save(os.path.join(path, img_file))
    with open(os.path.join(path, 'pos.csv'), 'a') as pos_fh:
        pos_fh.write(f'{img_file},{pip_pos[0]:0.2g},{pip_pos[1]:0.2g},{pip_pos[2]:0.2g},'
                     f'{image_stats["snr"]}\n')



class TrainingDataGenerator:
    def __init__(self, queue, data_args):
        self.queue = queue
        self.data_args = data_args
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            data = make_training_data(**self.data_args)
            self.queue.put(data)



if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser(description='Generate pipette detection training data files')
    parser.add_argument('--path', default="training", type=str, help='path to store training data')
    parser.add_argument('--size', default=1, type=int, help='number of training examples to generate')
    parser.add_argument('--difficulty', default=0, type=float, help='difficulty (0-1) controls signal/noise ratio, pipette focus and positioning')
    args = parser.parse_args()

    pos_file = os.path.join(args.path, 'pos.csv')
    if os.path.exists(pos_file):
        last_line = open(pos_file).readlines()[-1]
        img_count = int(last_line.split(',')[0].split('.')[0]) + 1
    else:
        img_count = 0

    if img_count >= args.size:
        print('Already generated enough training data')
        exit()

    training_data_queue = Queue(20)
    training_data_args = {
        'size': 400,
        'template': PipetteTemplates(glob.glob('template_data/template_*.npz')),
        'noise_data': NoiseData(glob.glob('template_data/background_data/ImageSequence*/image_000.ma')),
        'difficulty': args.difficulty,
    }
    threads = [TrainingDataGenerator(training_data_queue, training_data_args) for _ in range(8)]

    for i in tqdm(range(img_count, args.size)):
        data = training_data_queue.get()
        if np.isfinite(data[2]['snr']):
            save_training_data(args.path, i, *data)
