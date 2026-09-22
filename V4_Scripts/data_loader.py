import os, random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATASETS_RESIZED")
EXT = (".jpg", ".jpeg", ".png")
SEED = 1337


def tf(aug=False):
    t = [transforms.ToTensor()]
    if aug:
        t = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] + t
    return transforms.Compose(t)


class EyePACS(Dataset):
    def __init__(self, root, split, transform):
        self.transform = transform
        self.samples = []
        for lbl in range(5):
            d = os.path.join(root, split, str(lbl))
            for f in os.listdir(d):
                if f.lower().endswith(EXT):
                    self.samples.append((os.path.join(d, f), lbl))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y, "eyepacs"


class RFMid(Dataset):
    CSV = {"train": "RFMiD_Training_Labels.csv", "val": "RFMiD_Validation_Labels.csv",
           "test": "RFMiD_Testing_Labels.csv"}

    def __init__(self, root, split, transform, exclude_armd=False):
        self.transform = transform
        df = pd.read_csv(os.path.join(root, self.CSV[split]))
        df.columns = df.columns.str.lstrip("\ufeff").str.strip()
        df["ID"] = df["ID"].astype(str).str.strip()
        look = {r["ID"]: r for _, r in df.iterrows()}
        self.samples = []
        img_dir = os.path.join(root, split)
        for f in os.listdir(img_dir):
            if not f.lower().endswith(EXT): continue
            stem = os.path.splitext(f)[0]
            if stem not in look: continue
            r = look[stem]
            if exclude_armd and int(r["ARMD"]) == 1: continue
            self.samples.append((os.path.join(img_dir, f), int(r["Disease_Risk"])))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y, "rfmid"


class ORIGIA(Dataset):
    CSV = {"train": "ORIGIA_train.csv", "test": "ORIGIA_test.csv"}

    def __init__(self, root, split, transform):
        self.transform = transform
        df = pd.read_csv(os.path.join(root, self.CSV[split]))
        df["stem"] = df["ImageName"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        look = dict(zip(df["stem"], df["glaucoma"].astype(int)))
        self.samples = []
        img_dir = os.path.join(root, split)
        for f in os.listdir(img_dir):
            if f.lower().endswith(EXT):
                stem = os.path.splitext(f)[0]
                if stem in look:
                    self.samples.append((os.path.join(img_dir, f), look[stem]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y, "origia"


class ARMD(Dataset):
    # ARMD vs everything-else, RFMid only, all splits combined
    CSV = {"train": "RFMiD_Training_Labels.csv", "val": "RFMiD_Validation_Labels.csv",
           "test": "RFMiD_Testing_Labels.csv"}

    def __init__(self, root, transform):
        self.transform = transform
        self.samples = []
        for split, csv in self.CSV.items():
            img_dir = os.path.join(root, split)
            cpath = os.path.join(root, csv)
            if not (os.path.isdir(img_dir) and os.path.isfile(cpath)): continue
            df = pd.read_csv(cpath)
            df.columns = df.columns.str.lstrip("\ufeff").str.strip()
            df["ID"] = df["ID"].astype(str).str.strip()
            look = {r["ID"]: r for _, r in df.iterrows()}
            for f in os.listdir(img_dir):
                if not f.lower().endswith(EXT): continue
                stem = os.path.splitext(f)[0]
                if stem in look:
                    self.samples.append((os.path.join(img_dir, f), int(look[stem]["ARMD"])))

    def pos(self): return [i for i, (_, y) in enumerate(self.samples) if y == 1]
    def neg(self): return [i for i, (_, y) in enumerate(self.samples) if y == 0]
    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y, "rfmid"


def loader(ds, shuffle, bs, nw):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=True, persistent_workers=(nw > 0), drop_last=shuffle)


# Pretraining: all 3 datasets, ARMD held out.
# Signature matches the original: (batch_size, num_workers, split).
def get_combined_loader(batch_size=32, num_workers=6, split="train"):
    aug = split == "train"
    ds = [EyePACS(os.path.join(ROOT, "Eyepacs"), split, tf(aug)),
          RFMid(os.path.join(ROOT, "RFMid"), split, tf(aug), exclude_armd=True)]
    if split in ("train", "test"):
        ds.append(ORIGIA(os.path.join(ROOT, "ORIGIA"), split, tf(aug)))
    return loader(ConcatDataset(ds), aug, batch_size, num_workers)


# Locked ARMD split, identical across all models
def armd_splits(n_test_pos=50, seed=SEED):
    full = ARMD(os.path.join(ROOT, "RFMid"), tf(False))
    rng = random.Random(seed)
    pos, neg = full.pos(), full.neg()
    rng.shuffle(pos); rng.shuffle(neg)
    n = min(n_test_pos, len(pos) // 2)
    test = pos[:n] + neg[:n]
    return test, pos[n:], neg[n:]


def armd_probe(bs=32, nw=4):
    test, rp, rn = armd_splits()
    rng = random.Random(SEED + 1)
    train = rp + rng.sample(rn, min(len(rp), len(rn)))   # balanced
    full = ARMD(os.path.join(ROOT, "RFMid"), tf(False))
    return loader(Subset(full, train), True, bs, nw), loader(Subset(full, test), False, bs, nw)


def armd_finetune(slice_pos, repeat_seed=0, bs=16, nw=4):
    test, rp, rn = armd_splits()
    rng = random.Random(SEED + 100 + repeat_seed)
    rng.shuffle(rp); rng.shuffle(rn)
    n = min(slice_pos, len(rp))
    train = rp[:n] + rn[:n]
    full_aug = ARMD(os.path.join(ROOT, "RFMid"), tf(True))
    full = ARMD(os.path.join(ROOT, "RFMid"), tf(False))
    return loader(Subset(full_aug, train), True, bs, nw), loader(Subset(full, test), False, bs, nw)