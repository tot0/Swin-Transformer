# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import io
import os
import time
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
import numpy as np

from .zipreader import is_zip_path, ZipReader


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_np(dir, class_to_idx, extensions):
    images = []
    labels = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    labels.append(class_to_idx[target])
                    images.append(path)

    return np.asarray(images), np.asarray(labels)


def make_dataset_with_ann(ann_file, img_prefix, extensions):
    images = []
    with open(ann_file, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no", use_zip=True):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            # samples = make_dataset(root, class_to_idx, extensions)
            images, labels = make_dataset_np(root, class_to_idx, extensions)
            samples = images
            print(f'global_rank {dist.get_rank()} {images.dtype = } {labels.dtype = }')
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        #self.labels = [y_1k for _, y_1k in samples]
        self.images = images
        self.labels = labels
        self.classes = list(set(self.labels))
        
        self.world_size = dist.get_world_size()

        self.transform = transform
        self.target_transform = target_transform

        self.use_zip = use_zip
        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # if self.cache_mode == "part":
        #     n_subset = n_sample // world_size
        #     samples_bytes = [None for _ in range(n_subset)]
        #     samples_labels = [None for _ in range(n_subset)]
        # else:
        samples_bytes = [None for _ in range(n_sample)]
        samples_labels = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample // 10) == 0:
                t = time.time() - start_time
                print(f'global_rank {dist.get_rank()} cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            #path, target = self.samples[index]
            path = self.images[index]
            target = self.labels[index]
            if self.cache_mode == "full":
                if self.use_zip:
                    # samples_bytes[index] = (ZipReader.read(path), target)
                    samples_bytes[index] = ZipReader.read(path)
                    samples_labels[index] = target
                else:
                    with open(path, 'rb') as f:
                        #samples_bytes[index] = (f.read(), target)
                        samples_bytes[index] = f.read()
                    samples_labels[index] = target
            elif self.cache_mode == "part" and index % world_size == global_rank:
                if self.use_zip:
                    #samples_bytes[index] = (ZipReader.read(path), target)
                    samples_bytes[index // world_size] = ZipReader.read(path)
                    samples_labels[index // world_size] = target
                else:
                    with open(path, 'rb') as f:
                        #samples_bytes[index] = (f.read(), target)
                        samples_bytes[index // world_size] = f.read()
                    samples_labels[index // world_size] = target
            else:
                #samples_bytes[index] = (path, target)
                samples_bytes[index] = path
                samples_labels[index] = target
        #self.samples = samples_bytes
        self.images = np.asarray(samples_bytes)
        self.labels = np.asarray(samples_labels)

        print(f'global_rank {dist.get_rank()} {self.images.dtype = } {self.labels.dtype = }')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.cache_mode == "part":
            index = index // self.world_size

        #path, target = self.samples[index]
        path = self.images[index]
        target = self.labels[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CachedImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", use_zip=True):
        super(CachedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                                ann_file=ann_file, img_prefix=img_prefix,
                                                transform=transform, target_transform=target_transform,
                                                cache_mode=cache_mode, use_zip=use_zip)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.cache_mode == "part":
            index = index // self.world_size

        #path, target = self.samples[index]
        path = self.images[index]
        target = self.labels[index]

        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
