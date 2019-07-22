import numpy as np


class ClassBalancedBatchSampler(object):
    """
    BatchSampler that ensures a fixed amount of images per class are sampled in the minibatch
    """
    def __init__(self, targets, batch_size, images_per_class, ignore_index=None):
        self.targets = targets
        self.batch_size = batch_size
        self.images_per_class = images_per_class
        self.ignore_index = ignore_index
        self.reverse_index, self.ignored = self._build_reverse_index()

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample_batch()

    def _build_reverse_index(self):
        reverse_index = {}
        ignored = []
        for i, target in enumerate(self.targets):
            if target == self.ignore_index:
                ignored.append(i)
                continue
            if target not in reverse_index:
                reverse_index[target] = []
            reverse_index[target].append(i)
        return reverse_index, ignored

    def sample_batch(self):
        # Real batch size is self.images_per_class * (self.batch_size // self.images_per_class)
        num_classes = self.batch_size // self.images_per_class
        sampled_classes = np.random.choice(self.reverse_index.keys(),
                                           num_classes,
                                           replace=False)
        sampled_indices = []
        for cls in sampled_classes:
            # Need replace = True for datasets with non-uniform distribution of images per class
            sampled_indices.extend(np.random.choice(self.reverse_index[cls],
                                                    self.images_per_class,
                                                    replace=True))
        return sampled_indices

    def __len__(self):
        return len(self.targets) // self.batch_size
