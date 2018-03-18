import os
import torch
import torch.utils.data as data
import numpy as np
from torchvision.transforms import Normalize, CenterCrop
import PIL.Image as Image

def default_loader(path, length):
    files = os.listdir(path)
    files = sorted(files, key=lambda x:int(x.split('.')[0]))
    images = []
    cnt = 0
    for idx, filename in enumerate(files):
        if cnt > length - 1:
            break
        img = Image.open(path + '/' + filename).convert("RGB")
        images.append(np.asarray(img) / 255.0)
        cnt += 1
    return np.array(images)

def transform(fea):
    return torch.Tensor(fea).permute(3, 0, 1, 2)

class KTH(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None,
                 loader=default_loader, train=True, seq_len=20):
        fh = open(label)
        videos = []
        for line in fh.readlines():
            video_id = line.strip()
            videos.append(video_id)
        self.root = root
        self.videos = videos
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.length = seq_len

    def __getitem__(self, index):
        fn = self.videos[index]
        fea = self.loader(os.path.join(self.root, fn), self.length)
        if self.transform is not None:
            fea = self.transform(fea)
        return fea
        # return fea

    def __len__(self):
        return len(self.videos)

if __name__ == "__main__":
# torch.utils.data.DataLoader
    dataset = videoDataset(root="/home/xuchengming/figure_skating/frames",
                   label="./videos.txt", transform=transform)
    videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=1, shuffle=True, num_workers=2)

    for i, features in enumerate(videoLoader):
        import pdb;pdb.set_trace()
        print((i, len(labels)))
