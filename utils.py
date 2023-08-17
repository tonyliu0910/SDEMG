import os
import math
from scipy import signal

def check_path(path):
    # Check if path directory exists. If not, create a file directory
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    # Check if the folder of path exists
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)


def get_filepaths(directory,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def creat_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def resample(x, fs, fs_2):
    # x needs to be an 1D numpy array
    return signal.resample(x,int(x.shape[0]/fs * fs_2))