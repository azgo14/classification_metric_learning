from abc import ABCMeta, abstractmethod, abstractproperty
from PIL import Image

class Dataset(object):
    """
    This abstract class defines the interface needed for dataset loading
    All concrete subclasses need to implement the following method/property:

    def name: returns the name of the dataset

    def image_root_dir: the root directory of the images

    def _load: the actual logic to load the dataset,
        It needs to populate these three lists
        self.image_paths = []
        self.class_labels = []
        self.instance_labels = []
    """

    __metaclass__ = ABCMeta

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.class_labels = []
        self.instance_labels = []

        self._load()

        # Check dataset is loaded properly
        assert(len(self.image_paths) != 0)
        assert(len(self.instance_map) != 0)
        assert(len(self.image_paths) == len(self.instance_labels))
        assert(len(self.image_paths) == len(self.class_labels))

    @abstractproperty
    def name(self):
        raise NotImplementedError()

    @abstractproperty
    def image_root_dir(self):
        raise NotImplementedError()

    @abstractmethod
    def _load(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        im_path = self.image_paths[index]
        im = Image.open(im_path).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        class_target = self.class_labels[index]
        instance_target = self.instance_labels[index]
        return im, class_target, instance_target, index

    def __len__(self):
        return len(self.image_paths)
