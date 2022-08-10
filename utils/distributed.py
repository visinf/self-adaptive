import os
import torch
import torch.distributed

def init_process(opts,
                 gpu: int) -> int:

    # Define world size
    opts.world_size = opts.gpus
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    # Calculate rank
    rank = gpu

    # Initiate process group
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         world_size=opts.world_size,
                                         rank=rank)

    print(f"{rank + 1}/{opts.world_size} process initialized.\n")

    return rank

def clean_up():
    torch.distributed.destroy_process_group()
