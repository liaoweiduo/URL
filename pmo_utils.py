import numpy as np

from config import args


class Pool:
    """
    Pool stored class samples for the current clustering.

    A class instance contains (a set of image samples, class_label, class_label_str).
    """
    def __init__(self, capacity=8, max_num_classes=10, max_num_images=20):
        """
        :param capacity: Number of clusters. Typically 8 columns of classes.
        :param max_num_classes: Maximum number of classes can be stored in each cluster.
        :param max_num_images: Maximum number of images for each class.
        """
        self.capacity = capacity
        self.max_num_classes = max_num_classes
        self.max_num_images = max_num_images
        self.clusters = [[] for _ in range(self.capacity)]
        self.init()

    def init(self):
        self.clusters = [[] for _ in range(self.capacity)]

    def store(self, file_path):
        """
        Store pool to npy file.
        Only class information is stored.
        """
        pass

    def restore(self, file_path):
        """
        Restore pool from npy file.
        """
        self.init()
        np.load(file_path)

    def visualization(self):
        """
        Visualize the current pool with pool_montage.
        """
        pass

    def put(self, ):
        """
        Put class samples into clusters.
        Issues to handle:
            Maximum number of classes.
            Same class.
        """
        pass