import os.path
import random
from scipy.io import loadmat

from dataset import Dataset


class Cars196(Dataset):
    def __init__(self, root, train=True, transform=None, benchmark=True):
        self.meta_file = 'cars_annos.mat'
        self.benchmark = benchmark
        super(Cars196, self).__init__(root, train, transform)
        print "Loaded {} samples for dataset {},  {} classes, {} instances".format(len(self), self.name, self.num_cls, self.num_instance)

    @property
    def name(self):
        return 'cars196_{}_{}'.format('benchmark' if self.benchmark else 'random', 'train' if self.train else 'test')

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

        meta_data = loadmat(os.path.join(self.root, self.meta_file), squeeze_me=True)
        self.instance2class = []
        self.instance_names = {}
        self.class_names = {}

        # load train/test split
        instance_id_to_load = self._load_split(meta_data)

        self.class_map = {}
        self.instance_map = {}

        annotations = meta_data['annotations']
        for entry in annotations:
            # convert Matlab 1-indexing to python 0-indexing
            instance_id = int(entry['class']) - 1
            if str(instance_id) not in instance_id_to_load:
                continue

            class_name = meta_data['class_names'][instance_id]
            # The annotations are typically with the format of (Make, Model, Type, Year)
            make = class_name.split(' ')[0]
            type = class_name.split(' ')[-2]
            vehicle_type = ' '.join([make, type])
            if vehicle_type not in self.class_map:
                self.class_map[vehicle_type] = len(self.class_map)

            self.class_labels.append(self.class_map[vehicle_type])
            self.class_names[self.class_map[vehicle_type]] = vehicle_type

            self.image_paths.append(os.path.join(self.image_root_dir, entry['relative_im_path']))
            # consolidate the ids into continuous labels
            if instance_id not in self.instance_map:
                self.instance2class.append(self.class_map[vehicle_type])
                self.instance_map[instance_id] = len(self.instance_map)

            self.instance_labels.append(self.instance_map[instance_id])
            self.instance_names[self.instance_map[instance_id]] = class_name

    def _load_split(self, meta_data, benchmark=True):
        split_file = 'cars_{}_{}_cls_split.txt'.format('benchmark' if benchmark else 'random',
                                                       'train' if self.train else 'test')
        split = os.path.join(self.root, split_file)
        if not os.path.exists(split):
            # split the classes into 50:50 train:test split

            num_total_classes = meta_data['class_names'].size
            shuffled_idxs = range(num_total_classes)

            if not benchmark:
                import random
                random.seed(2018)
                random.shuffle(shuffled_idxs)

            with open(os.path.join(self.root,
                                   'cars_{}_train_cls_split.txt'.format('benchmark' if benchmark else 'random'))
                    , 'wb') as wf:
                for i in shuffled_idxs[:num_total_classes//2]:
                    wf.write(str(i) + '\n')

            with open(os.path.join(self.root,
                                   'cars_{}_test_cls_split.txt'.format('benchmark' if benchmark else 'random'))
                    , 'wb') as wf:
                for i in shuffled_idxs[num_total_classes//2:]:
                    wf.write(str(i) + '\n')

        with open(split) as f:
            lines = f.readlines()

        return set([x.strip() for x in lines])


if __name__ == '__main__':

    cars196_train_set = Cars196('/data1/data/cars196')
    print "Loaded {} samples for dataset {}".format(len(cars196_train_set), cars196_train_set.name)
    for i in random.sample(range(0,len(cars196_train_set)), 5):
        image_id, class_id, instance_id = cars196_train_set[i]
        print "Image  {} has label {}, name {}, label {}, name {}".format(cars196_train_set.image_paths[i], instance_id,
                                                                          cars196_train_set.instance_names[instance_id],
                                                                          class_id,
                                                                          cars196_train_set.class_names[class_id])

    cars196_test_set = Cars196('/data1/data/cars196', train=False)
    print "Loaded {} samples for dataset {}".format(len(cars196_test_set), cars196_test_set.name)
    for i in [2022, 1668, 2041, 1710, 2233, 2160, 3970, 3800]:
    # for i in random.sample(range(0,len(cars196_train_set)), 5):
        image_id, class_id, instance_id = cars196_test_set[i]

        print "Image  {} has label {}, name {}, label {}, name {}".format(cars196_test_set.image_paths[i], instance_id,
                                                                      cars196_test_set.instance_names[instance_id],
                                                                      class_id, cars196_test_set.class_names[class_id])

    for i, c in enumerate(cars196_test_set.instance2class):
        print "Instace name {} has class name {}".format(cars196_test_set.instance_names[i],
                                                         cars196_test_set.class_names[c])
