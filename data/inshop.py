import csv
import os.path
import random

from dataset import Dataset


class InShop(Dataset):
    def __init__(self, root, train=True, query=True, transform=None):
        self.query = query
        self.split_file = "list_eval_partition.txt"
        super(InShop, self).__init__(root, train, transform)
        print "Loaded {} samples for dataset {},  {} classes, {} instances".format(len(self),
                                                                                   self.name,
                                                                                   self.num_cls,
                                                                                   self.num_instance)
    @property
    def name(self):
        return 'inshop_{}_{}'.format('train' if self.train else 'test',
                                     'query' if self.query else 'gallery')
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
        self.class_map = {}
        with open(os.path.join(self.root, self.split_file), 'r') as f:
            for line in f.read().splitlines()[2:]:
                image_name, item_id, evaluation_status = line.strip().split()
                skip = True
                if self.train:
                    if evaluation_status == "train":
                        # Train data points
                        self.image_paths.append(os.path.join(self.image_root_dir, image_name))
                        class_id = item_id
                        if class_id not in self.class_map:
                            self.class_map[class_id] = len(self.class_map)
                        self.class_labels.append(self.class_map[class_id])
                else:
                    if evaluation_status != "train":
                        # Test data points

                        # Keep class ids consistent amongst query and gallery data points. The class id set is
                        # the same for query and gallery.
                        class_id = item_id
                        if class_id not in self.class_map:
                            self.class_map[class_id] = len(self.class_map)

                        if evaluation_status == "query" and self.query:
                            self.image_paths.append(os.path.join(self.image_root_dir, image_name))
                            self.class_labels.append(self.class_map[class_id])
                        elif evaluation_status == "gallery" and not self.query:
                            self.image_paths.append(os.path.join(self.image_root_dir, image_name))
                            self.class_labels.append(self.class_map[class_id])

        # same thing for in-shop
        self.instance_labels = self.class_labels
        self.instance_map = self.class_map


if __name__ == '__main__':
    inshop_train_set = InShop('/data1/data/inshop')
    for i in random.sample(range(0,len(inshop_train_set)), 5):
        image_id, class_id, instance_id, idx = inshop_train_set[i]
        assert idx == i
        print "Image  {} has class label {}, instance label {}".format(inshop_train_set.image_paths[i],
                                                                       instance_id,
                                                                       class_id)

    inshop_query_set = InShop('/data1/data/inshop', train=False, query=True)
    for i in random.sample(range(0,len(inshop_query_set)), 5):
        image_id, class_id, instance_id, idx = inshop_query_set[i]
        assert idx == i
        print "Image  {} has class label {}, instance label {}".format(inshop_query_set.image_paths[i],
                                                                       instance_id,
                                                                       class_id)

    inshop_index_set = InShop('/data1/data/inshop', train=False, query=False)
    for i in random.sample(range(0,len(inshop_index_set)), 5):
        image_id, class_id, instance_id, idx = inshop_index_set[i]
        assert idx == i
        print "Image  {} has class label {}, instance label {}".format(inshop_index_set.image_paths[i],
                                                                       instance_id,
                                                                       class_id)
