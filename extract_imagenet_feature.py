import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import lmdb
from datasets import ImageNet
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import time
import os

# Seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='/home/ubuntu/data/datasets/imgnet1k_comp/avif/80')
parser.add_argument('--output_path', default='/mnt/tmpfs/latents/avif/80')
parser.add_argument('--device_ids', nargs=1, type=str)
args = parser.parse_args()
args.device_ids = [int(i) for i in args.device_ids[0].split(',')]

resolution = 256
random_flip = True
device_ids = [0,1]

device = torch.device("cuda") if torch.cuda.is_available() and device_ids else torch.device("cpu")
model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
model.eval()


if device_ids is not None:
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(f"cuda:{device_ids[0]}")


@torch.no_grad()
def test():
    img = torch.rand([64, 3, 256, 256], dtype=torch.float32)
    img = img.to(device)
    for i in range(10): img1 = model(img, fn='encode_moments')

    t = time.time()
    for i in range(20): img1 = model(img, fn='encode_moments')
    t1 = time.time()
    dt1 = t1 - t

    model2 = torch.compile(model)
    for i in range(10): img2 = model2(img)
    
    t = time.time
    for i in range(20): img2 = model2(img)
    t1 = time.time()
    dt2 = t1 - t

    diff = img1 - img2
    diff = torch.abs(diff)
    print(torch.abs(diff), torch.min(diff), torch.max(diff))
    print(dt1, dt2)
    # failed to coimpile

train_dataset = ImageNet(path=args.input_path, resolution=resolution, random_flip=False).train

train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=False,
                                  num_workers=8, pin_memory=True, persistent_workers=True)


@torch.cuda.amp.autocast()
def generate_moments(batch):
    img, label = batch
    # Apply random flip if needed
    if random_flip:
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        label = torch.cat([label, label], dim=0)
    img = img.to(device)
    moments = model(img, fn='encode_moments')
    moments = moments.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    return moments, label


def write_to_lmdb(env, key, moment:np.ndarray, label:np.ndarray):
    with env.begin(write=True) as txn:
        # Write moment and label to LMDB separately
        moment, label = moment.astype(np.float16), label.astype(np.int64)
        txn.put(f"{key}_moment".encode('ascii'), moment.tobytes())
        txn.put(f"{key}_label".encode('ascii'), label.tobytes())


def read_from_lmdb(env, key):
    with env.begin(write=False) as txn:
        moment_data = txn.get(f"{key}_moment".encode('ascii'))
        label_data = txn.get(f"{key}_label".encode('ascii'))
    # Convert bytes back to numpy arrays
    moment: np.ndarray = np.frombuffer(moment_data, dtype=np.float16)
    label: np.ndarray = int(np.frombuffer(label_data, dtype=np.int64))
    return moment, label

@torch.no_grad()
def generate():
    global idx
    idx = 0
    # Create LMDB environment
    map_size = 109951162776 * 2  # 2TB
    os.makedirs(args.output_path, exist_ok=True)
    env = lmdb.open(args.output_path, map_size=map_size)
    
    for batch in tqdm(train_dataset_loader, leave=False): 
        moments, labels = generate_moments(batch)
        for moment, lb in zip(moments, labels):
            # Write moment and label to LMDB
            key = f"{idx:08d}"
            write_to_lmdb(env, key, moment, np.array(lb, dtype=np.int32))
            idx += 1
    print(f'saved {idx} files')

@torch.no_grad()
def verify():
    global idx
    idx = 0
    # Open LMDB environment
    env = lmdb.open(args.output_path, readonly=True, lock=False)
    
    for batch in tqdm(train_dataset_loader, leave=False):
        moments, labels = generate_moments(batch)
        for moment, lb in zip(moments, labels):
            # Read moment and label from LMDB
            key = f"{idx:08d}"
            fetched_moment, fetched_lb = read_from_lmdb(env, key)
            fetched_moment = fetched_moment.reshape([-1, resolution // 8, resolution // 8])
            
            # Verify integrity
            if not (np.allclose(moment, fetched_moment) and lb == fetched_lb):
                print(f"Verification failed for index {idx}")
            else:
                idx += 1
    print(f'verified {idx} files')


def main():
    # test()
    generate()
    verify()



if __name__ == "__main__":
    main()