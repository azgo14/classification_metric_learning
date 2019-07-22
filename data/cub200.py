import os.path
import random
import csv

from dataset import Dataset


class Cub200(Dataset):
    def __init__(self, root, train=True, transform=None, benchmark=True):
        self.image_id_2_relfile = os.path.join(root, 'images.txt')
        self.image_id_2_cls_id_file = os.path.join(root, 'image_class_labels.txt')
        self.class_name_file = os.path.join(root, 'classes.txt')
        self.benchmark = benchmark
        super(Cub200, self).__init__(root, train, transform)
        print "Loaded {} samples for dataset {},  {} classes, {} instances".format(len(self), self.name, self.num_cls, self.num_instance)

    @property
    def name(self):
        return 'cub200_{}_{}'.format('benchmark' if self.benchmark else 'random', 'train' if self.train else 'test')

    @property
    def image_root_dir(self):
        return os.path.join(self.root, 'images')

    @property
    def num_cls(self):
        return len(self.class_map)

    @property
    def num_instance(self):
        return len(self.instance_map)

    def _load(self):

        meta_data = self._load_meta_data()
        self.instance_names = {}

        # load train/test split
        instance_id_to_load = self._load_split(meta_data, benchmark=self.benchmark)

        self.class_map = {}
        self.instance_map = {}

        for image_id, instance_id in meta_data['id2cls'].items():

            if str(instance_id) not in instance_id_to_load:
                continue

            self.image_paths.append(os.path.join(self.image_root_dir, meta_data['id2file'][image_id]))
            # consolidate the ids into continuous labels from 0 to num_instance
            if instance_id not in self.instance_map:
                self.instance_map[instance_id] = len(self.instance_map)
            self.instance_labels.append(self.instance_map[instance_id])
            self.instance_names[self.instance_map[instance_id]] = meta_data['class_names'][instance_id]

            # Set the class_id to instance id for now
            if instance_id not in self.class_map:
                self.class_map[instance_id] = len(self.class_map)
            self.class_labels.append(self.class_map[instance_id])

    def _load_meta_data(self):
        # all the ids are 1-indexed. convert them into 0-indexed
        meta_data = {}
        meta_data['id2file'] = {}
        meta_data['id2cls'] = {}
        meta_data['class_names'] = {}
        with open(self.image_id_2_relfile) as rf:
            csvreader = csv.reader(rf, delimiter=' ')
            for row in csvreader:
                meta_data['id2file'][int(row[0])] = row[1]

        with open(self.image_id_2_cls_id_file) as rf:
            csvreader = csv.reader(rf, delimiter=' ')
            for row in csvreader:
                meta_data['id2cls'][int(row[0])] = int(row[1])

        with open(self.class_name_file) as rf:
            csvreader = csv.reader(rf, delimiter=' ')
            for row in csvreader:
                meta_data['class_names'][int(row[0])] = row[1]

        self.class_names = meta_data['class_names']
        return meta_data

    def _load_split(self, meta_data, benchmark=True):
        split_file = 'cub_{}_{}_cls_split.txt'.format('benchmark' if benchmark else 'random',
                                                      'train' if self.train else 'test')
        split = os.path.join(self.root, split_file)
        if not os.path.exists(split):
            # split the classes into 50:50 train:test split

            num_total_classes = len(meta_data['class_names'])
            shuffled_idxs = range(num_total_classes)

            if not benchmark:
                import random
                random.seed(2018)
                random.shuffle(shuffled_idxs)

            with open(os.path.join(self.root,
                                   'cub_{}_train_cls_split.txt'.format('benchmark' if benchmark else 'random'))
                    , 'wb') as wf:
                for i in shuffled_idxs[:num_total_classes//2]:
                    # make the class id 1-indexed to be consistent with the dataset
                    wf.write(str(i+1) + '\n')

            with open(os.path.join(self.root,
                                   'cub_{}_test_cls_split.txt'.format('benchmark' if benchmark else 'random'))
                    , 'wb') as wf:
                for i in shuffled_idxs[num_total_classes//2:]:
                    # make the class id 1-indexed to be consistent with the dataset
                    wf.write(str(i+1) + '\n')

        with open(split) as f:
            lines = f.readlines()

        return set([x.strip() for x in lines])


if __name__ == '__main__':

    train_set = Cub200('/data1/data/cub200/CUB_200_2011')
    for i in random.sample(range(0,len(train_set)), 5):
        image_id, class_id, instance_id = train_set[i]
        print "Image  {} has label {}, name {}".format(train_set.image_paths[i], instance_id, train_set.instance_names[instance_id])

    test_set = Cub200('/data1/data/cub200/CUB_200_2011', train=False)
    for i in random.sample(range(0,len(test_set)), 5):
        image_id, class_id, instance_id = test_set[i]
        print "Image  {} has label {}, name {}".format(test_set.image_paths[i], instance_id, test_set.instance_names[instance_id])

