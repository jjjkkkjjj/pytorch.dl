import torch

from .._utils import _check_ins

def ohem(confs, hard_sample_nums, random_sample_nums=None):
    """
    :param confs: Tensor, shape = (num, 1)
    :param dim: int, confs will be sorted with descending order along given dim
    :param hard_sample_nums: int
    :param random_sample_nums: int or None, if it's None, random sampling will not be done
    :returns:
        hard_indices: Long Tensor, shape = (num, 1)
        rand_indices: Long Tensor, shape = (num, 1). Note: if random indices doesn't exist, return empty long tensor
        sample_nums: int, total sampling number
        ~~Usage~~

        if rand_indices.numel() > 0:
            ~~~
    """
    assert confs.ndim == 2 and confs.numel() == confs.shape[0], "confs must be 2-dim with (num, 1) "
    hard_sample_nums = _check_ins('hard_sample_nums', hard_sample_nums, int, allow_none=False)
    random_sample_nums = _check_ins('random_sample_nums', random_sample_nums, int, allow_none=True, default=None)

    device = confs.device
    # hard sampling
    _, indices = torch.sort(confs, dim=0, descending=True)
    hard_indices = indices[:hard_sample_nums]

    # random sampling
    indices = indices[hard_sample_nums:]
    if indices.numel() > 0 and random_sample_nums is not None:
        # permute indices order randomly
        _rand_inds = torch.randperm(indices.numel()).unsqueeze(-1)
        indices = indices[_rand_inds]
        rand_indices = indices[:random_sample_nums]
    else:
        # empty indices
        rand_indices = torch.tensor([], dtype=torch.long, device=device)

    return hard_indices, rand_indices, hard_indices.numel() + rand_indices.numel()