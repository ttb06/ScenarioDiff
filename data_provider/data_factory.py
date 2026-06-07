from data_provider.data_loader import Dataset_Custom
from functools import partial
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}

def _seed_worker(worker_id, seed):
    worker_seed = (seed + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = True if flag == 'train' else False
    drop_last = True if flag == 'train' else False
    
    if flag == 'train' or flag == 'valid':
        batch_size = args.batch_size
    else:
        batch_size = args.eval_batch_size or args.batch_size

    
    freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        text_len=args.text_len,
        use_intrinsic=args.with_intrinsic,
        use_future_context=args.with_future_hint,
        text_perturb_ratio=getattr(args, "text_perturb_ratio", 0.0),
        text_perturb_targets=getattr(args, "text_perturb_targets", None),
        text_perturb_seed=getattr(args, "text_perturb_seed", getattr(args, "seed", 2025)),
    )
    print(flag, len(data_set))
    seed = int(getattr(args, "seed", 2025))
    flag_offset = {"train": 0, "valid": 10_000, "test": 20_000}[flag]
    generator = torch.Generator()
    generator.manual_seed(seed + flag_offset)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=partial(_seed_worker, seed=seed + flag_offset),
        generator=generator)
    return data_set, data_loader
