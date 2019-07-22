import csv
import os.path

from dataset import Dataset


class StanfordOnlineProducts(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.info_file = 'Ebay_{}.txt'.format('train' if train else 'test')
        super(StanfordOnlineProducts, self).__init__(root, train, transform)
        print "Loaded {} samples for dataset {},  {} classes, {} instances".format(len(self),
                                                                                   self.name,
                                                                                   self.num_cls,
                                                                                   self.num_instance)

    @property
    def name(self):
        return 'stanford_online_products_{}'.format('train' if self.train else 'test')

    @property
    def image_root_dir(self):
        return self.root

    @property
    def num_cls(self):
        return len(self.class_map)

    @property
    def num_instance(self):
        return len(self.instance_map)

    def _load(self):
        self.instance_map = {}
        self.class_map = {}
        with open(os.path.join(self.root, self.info_file), 'r') as f:
            reader = csv.DictReader(f, delimiter=' ')
            for entry in reader:
                self.image_paths.append(os.path.join(self.image_root_dir, entry['path']))
                class_id = int(entry['super_class_id']) - 1
                if class_id not in self.class_map:
                    self.class_map[class_id] = len(self.class_map)
                self.class_labels.append(self.class_map[class_id])

                instance_id = entry['class_id']
                if instance_id not in self.instance_map:
                    self.instance_map[instance_id] = len(self.instance_map)
                self.instance_labels.append(self.instance_map[instance_id])

