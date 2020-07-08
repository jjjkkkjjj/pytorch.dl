from torch.utils.data import Dataset
import abc

class _DatasetBase(Dataset):
    @property
    @abc.abstractmethod
    def class_nums(self):
        pass
    @property
    @abc.abstractmethod
    def class_labels(self):
        pass