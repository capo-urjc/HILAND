import math
import torch


class UniformBinarySampler:
    def __init__(self, labels, batch_size):
        self.labels = torch.tensor(labels)
        self.unique_classes, self.class_counts = torch.unique(self.labels, return_counts=True)
        self.max_count = torch.max(self.class_counts)
        self.num_classes = self.unique_classes.shape[0]
        self.batch_size = batch_size

    def __shuffle__(self):
        msg = 'Sampler: '
        class_indxs = {}
        for class_id, count in zip(self.unique_classes, self.class_counts):
            class_id = int(class_id)
            class_idx = torch.where(self.labels == class_id)[0]
            shuffled_class_idx = class_idx[torch.randperm(class_idx.shape[0])]
            shuffled_class_idx = shuffled_class_idx.repeat(int(math.ceil(self.max_count / shuffled_class_idx.shape[0])))
            shuffled_class_idx = shuffled_class_idx[:self.max_count]
            class_indxs[class_id] = shuffled_class_idx
            msg += 'class {}: {} - {}, '.format(class_id, shuffled_class_idx.shape[0],
                                                torch.unique(self.labels[class_idx]))

        print(msg)

        return class_indxs

    def __iter__(self):
        class_idx = self.__shuffle__()

        items_per_class = self.batch_size // self.num_classes
        left_over_items = self.batch_size - (items_per_class * self.num_classes)
        print('Sampler: items per class: {}'.format(items_per_class))
        batch = []
        for i in range(0, math.ceil(self.max_count / items_per_class)):
            for class_id in self.unique_classes:
                class_id = int(class_id)
                batch.append(class_idx[class_id][i * items_per_class: (i + 1) * items_per_class])

            yield torch.hstack(batch).tolist()
            batch = []

    def __len__(self):
        return ((self.max_count * self.num_classes) + self.batch_size - 1) // self.batch_size
