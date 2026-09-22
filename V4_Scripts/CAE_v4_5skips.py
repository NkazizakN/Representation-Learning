import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from data_loader import get_combined_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR   = "checkpoints/5skips"
CKPT_PATH  = os.path.join(CKPT_DIR, "last.pt")
BEST_PATH  = os.path.join(CKPT_DIR, "best_CAE_5skips.pt")
LR         = 1e-3
BATCH_SIZE = 64
EPOCHS     = 50
WANDB_KEY  = "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"


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Block 1: 3x512x512 -> 64x256x256
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Block 2: 64x256x256 -> 128x128x128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Block 3: 128x128x128 -> 256x64x64
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Block 4: 256x64x64 -> 512x32x32
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Block 5: 512x32x32 -> 512x16x16
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Block 6: 512x16x16 -> 512x8x8
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        s1 = self.block1(x)   # 64  x 256x256
        s2 = self.block2(s1)  # 128 x 128x128
        s3 = self.block3(s2)  # 256 x 64x64
        s4 = self.block4(s3)  # 512 x 32x32
        s5 = self.block5(s4)  # 512 x 16x16
        z  = self.block6(s5)  # 512 x 8x8
        return z, (s1, s2, s3, s4, s5)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Block 1: 512x8x8 -> 512x16x16, concat s5 (512) -> 3x3 conv -> 512
        self.block1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1    = nn.BatchNorm2d(512)
        self.act1   = nn.LeakyReLU(0.1, inplace=True)
        self.merge1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Block 2: 512x16x16 -> 512x32x32, concat s4 (512) -> 3x3 conv -> 512
        self.block2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2    = nn.BatchNorm2d(512)
        self.act2   = nn.LeakyReLU(0.1, inplace=True)
        self.merge2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Block 3: 512x32x32 -> 256x64x64, concat s3 (256) -> 3x3 conv -> 256
        self.block3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3    = nn.BatchNorm2d(256)
        self.act3   = nn.LeakyReLU(0.1, inplace=True)
        self.merge3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Block 4: 256x64x64 -> 128x128x128, concat s2 (128) -> 3x3 conv -> 128
        self.block4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4    = nn.BatchNorm2d(128)
        self.act4   = nn.LeakyReLU(0.1, inplace=True)
        self.merge4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Block 5: 128x128x128 -> 64x256x256, concat s1 (64) -> 3x3 conv -> 64
        self.block5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5    = nn.BatchNorm2d(64)
        self.act5   = nn.LeakyReLU(0.1, inplace=True)
        self.merge5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Block 6: 64x256x256 -> 3x512x512 — no skip, no BatchNorm, Sigmoid
        self.block6  = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, skips):
        s1, s2, s3, s4, s5 = skips

        x = self.act1(self.bn1(self.block1(z)))     # 512 x 16x16
        x = self.merge1(torch.cat([x, s5], dim=1)) # concat -> 1024, merge -> 512

        x = self.act2(self.bn2(self.block2(x)))     # 512 x 32x32
        x = self.merge2(torch.cat([x, s4], dim=1)) # concat -> 1024, merge -> 512

        x = self.act3(self.bn3(self.block3(x)))     # 256 x 64x64
        x = self.merge3(torch.cat([x, s3], dim=1)) # concat -> 512,  merge -> 256

        x = self.act4(self.bn4(self.block4(x)))     # 128 x 128x128
        x = self.merge4(torch.cat([x, s2], dim=1)) # concat -> 256,  merge -> 128

        x = self.act5(self.bn5(self.block5(x)))     # 64  x 256x256
        x = self.merge5(torch.cat([x, s1], dim=1)) # concat -> 128,  merge -> 64

        return self.sigmoid(self.block6(x))         # 3 x 512x512


class CAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z, skips = self.encoder(x)
        x_hat    = self.decoder(z, skips)
        return x_hat, z


criterion = nn.L1Loss()


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE)
        _, z = model(imgs)
        z_vec = z.mean(dim=(2, 3))
        embeddings.append(z_vec.cpu())
    return torch.cat(embeddings, dim=0)


def save_checkpoint(path, model, optimiser, scheduler, epoch, best_loss):
    torch.save({
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "best_loss": best_loss,
    }, path)


def load_checkpoint(path, model, optimiser, scheduler):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt["best_loss"]


def train():
    print("Loading dataloaders...", flush=True)
    train_loader = get_combined_loader(BATCH_SIZE, split="train")
    val_loader   = get_combined_loader(BATCH_SIZE, split="val")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    print("Building model...", flush=True)
    model     = CAE().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5
    )
    print("Model ready.", flush=True)

    start_epoch = 1
    best_loss   = float("inf")
    os.makedirs(CKPT_DIR, exist_ok=True)

    if os.path.exists(CKPT_PATH):
        print("Checkpoint found — loading...", flush=True)
        start_epoch, best_loss = load_checkpoint(CKPT_PATH, model, optimiser, scheduler)
        print(f"Resumed from epoch {start_epoch} | best_loss so far: {best_loss:.4f}", flush=True)
    else:
        print("No checkpoint found — starting fresh.", flush=True)

    print("Logging into wandb...", flush=True)
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="CAE-5skips",
        name=f"lr_{LR}_bs_{BATCH_SIZE}",
        config={"lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS}
    )
    print("Wandb initialised.", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} | lr={optimiser.param_groups[0]['lr']}", flush=True)

        model.train()
        train_loss = 0

        for imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True):
            imgs = imgs.to(DEVICE)
            optimiser.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train done | loss: {train_loss:.4f}", flush=True)

        print("Running validation...", flush=True)
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, _, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=True):
                imgs = imgs.to(DEVICE)
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         optimiser.param_groups[0]["lr"],
            "epoch":      epoch
        })

        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            with open(os.path.join(CKPT_DIR, "best_config.txt"), "w") as f:
                f.write(f"Best val loss: {best_loss:.6f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"lr: {LR}\n")
                f.write(f"batch_size: {BATCH_SIZE}\n")
            print(f"  --> New best: {best_loss:.4f} | model saved.", flush=True)

        save_checkpoint(CKPT_PATH, model, optimiser, scheduler, epoch, best_loss)
        print("Checkpoint saved.", flush=True)

    print("Training complete.", flush=True)
    wandb.finish()


if __name__ == "__main__":
    train()