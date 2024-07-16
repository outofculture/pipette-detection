import MetaArray
import numpy as np
import coorx
import scipy.ndimage


# (filename, pipette position (frame, row, col))

datasets = [
    {'file': '/mp0/d/data/autopatch_test/2024.02.22_000/slice_001/cell_005/video_000.ma', 
     'position': (40, 495, 527), 'angle': 104, 'crop': (slice(300, 700), slice(300, 700))}, # pipette
    {'file': '/mp0/d/data/autopatch_test/2024.02.22_000/slice_001/cell_003/video_000.ma', 
     'position': (12, 627, 526), 'angle': 104, 'crop': (slice(300, 700), slice(300, 700))}, # pipette at surface
    {'file': '/mp0/d/data/autopatch_test/2024.01.24_000/slice_002/cell_008/video_000.ma', 
     'position': (61, 766, 590), 'angle': 104, 'crop': (slice(400, 800), slice(0, 400))}, # pipette + nucleus
    {'file': '/mp0/d/data/autopatch_test/2024.01.24_000/slice_002/cell_008/video_001.ma', 
     'position': (22, 681, 440), 'angle': 104, 'crop': (slice(300, 700), slice(300, 700))}, # pipette + clog
    {'file': '/mp0/d/data/autopatch_test/2024.02.22_000/slice_001/cell_005/video_002.ma', 
     'position': (22, 503, 523), 'angle': 104, 'crop': (slice(300, 700), slice(300, 700))}, # noisy pipette
]


# load all datasets
def load_test_datasets():
    global datasets
    for ds in datasets:
        # load image
        ma = MetaArray.MetaArray(file=ds['file'])
        ds['data'] = ma
        img = ma.asarray().astype(float)

        # rotate and crop image
        rot, tr = make_rotated_crop(img, -ds['angle'], ds['crop'])
        # convert to 0-255 rgb
        rot = (rot - rot.min()) / (rot.max() - rot.min()) * 255
        rot = np.stack([rot] * 3, axis=-1)
        ds['rotated_data'] = rot
        ds['tr'] = tr

        # store rotated tip position
        pos = np.array(ds['position'][1:])
        ds['rotated_position'] = tr.map(pos)  # x, y

        # get z position relative to tip
        z_vals = ma._info[0]['translation'][:, 2]
        tip_z = z_vals[ds['position'][0]]
        ds['z'] = tip_z - z_vals

        print(ds['file'])

    return datasets



# Make transform mapping unrotated to rotated coordinates
def make_rotation_transform(angle, img1, img2, crop):
    center1 = np.array(img1.shape[-2:]) / 2
    center2 = np.array(img2.shape[-2:]) / 2
    tr = coorx.AffineTransform(dims=(2, 2))
    tr.translate(-center1)
    tr.rotate(-angle)
    tr.translate(center2)
    tr.translate([
        -crop[0].indices(img2.shape[-2])[0], 
        -crop[1].indices(img2.shape[-1])[0], 
    ])
    return tr


def make_rotated_crop(img, angle, crop):
    """Rotate and crop an image.
    Returns the rotated and cropped image and the transform mapping the original to the rotated 
    image (rows, cols).
    
    Parameters
    ----------
    img : ndarray
        The image to rotate and crop. Any ndim allowed, but the last two axes must be (rows, cols)
    angle : float   
        The angle in degrees to rotate the image.
    crop : tuple of slice
        The crop window to apply to the rotated image.
    """
    rotated_img = scipy.ndimage.rotate(img, angle, axes=(img.ndim-2, img.ndim-1), reshape=False)
    cropped_rotated_img = rotated_img[..., crop[0], crop[1]]
    tr = make_rotation_transform(angle, img, rotated_img, crop)
    return cropped_rotated_img, tr

