from torch.utils.data import Dataset
import abc

reapply_in_exception = True
maximum_reapply = 10

class _DatasetBase(Dataset):
    @property
    @abc.abstractmethod
    def class_nums(self):
        pass
    @property
    @abc.abstractmethod
    def class_labels(self):
        pass