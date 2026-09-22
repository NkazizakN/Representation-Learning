import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from torch_cluster import knn_graph
from data_loader import get_combined_loader

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR   = "checkpoints/gae_knn"
CKPT_PATH  = os.path.join(CKPT_DIR, "last.pt")
BEST_PATH  = os.path.join(CKPT_DIR, "best_gae_knn.pt")
LR         = 1e-3
BATCH_SIZE = 32
EPOCHS     = 50
PATCH      = 32      # 512/32 = 16x16 = 256 nodes per image
GRID       = 512 // PATCH
K          = 10      # k-NN neighbours (sweep: 5 / 10 / 20)
WANDB_KEY  = "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"


# ── Shallow conv stem: one patch -> one node feature vector ──
class PatchStem(nn.Module):
    def __init__(self, out_ch=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),          # collapse patch -> 1 vector
        )

    def forward(self, patches):               # (N, 3, PATCH, PATCH)
        return self.net(patches).flatten(1)   # (N, out_ch)


# ── Conv decoder: spatial latent (512 x GRID x GRID) -> image (3 x 512 x 512) ──
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # GRID=16 -> 32 -> 64 -> 128 -> 256 -> 512  (5 upsamples)
        def up(i, o, bn=True):
            layers = [nn.ConvTranspose2d(i, o, 3, stride=2, padding=1, output_padding=1)]
            if bn:
                layers += [nn.BatchNorm2d(o), nn.LeakyReLU(0.1, inplace=True)]
            return layers
        self.net = nn.Sequential(
            *up(512, 512), *up(512, 256), *up(256, 128),
            *up(128, 64),  *up(64, 3, bn=False), nn.Sigmoid(),
        )

    def forward(self, z):                      # (B, 512, GRID, GRID)
        return self.net(z)


class GAE(nn.Module):
    def __init__(self, latent_ch=512, k=K, mp_layers=3):
        super().__init__()
        self.k       = k
        self.grid    = GRID
        self.stem    = PatchStem(latent_ch)
        # EdgeConv message-passing layers (dynamic graph rebuilt each forward)
        self.mp = nn.ModuleList([
            EdgeConv(nn.Sequential(
                nn.Linear(2 * latent_ch, latent_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(latent_ch, latent_ch),
            )) for _ in range(mp_layers)
        ])
        self.decoder = Decoder()

    def to_patches(self, x):
        # (B,3,512,512) -> (B*grid*grid, 3, PATCH, PATCH) + batch index
        B = x.shape[0]
        p = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)      # B,3,grid,grid,P,P
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, PATCH, PATCH)
        batch = torch.arange(B, device=x.device).repeat_interleave(self.grid * self.grid)
        return p, batch, B

    def encode(self, x):
        patches, batch, B = self.to_patches(x)
        h = self.stem(patches)                      # (B*N, 512) node features
        for layer in self.mp:
            h_norm = F.normalize(h, dim=1)          # cosine via normalized Euclidean
            edge_index = knn_graph(h_norm, k=self.k, batch=batch, loop=False)
            h = h + layer(h, edge_index)            # residual message passing
        # scatter nodes back to grid -> spatial latent (B, 512, grid, grid)
        z = h.view(B, self.grid, self.grid, -1).permute(0, 3, 1, 2).contiguous()
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


criterion = nn.L1Loss()


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE)
        _, z = model(imgs)
        embeddings.append(z.mean(dim=(2, 3)).cpu())
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
    model     = GAE().to(DEVICE)
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
        project="GAE-knn",
        name=f"lr_{LR}_bs_{BATCH_SIZE}_k_{K}",
        config={"lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                "patch": PATCH, "k": K}
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
                val_loss += criterion(recon, imgs).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         optimiser.param_groups[0]["lr"],
            "epoch":      epoch,
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
                f.write(f"k: {K}\n")
            print(f"  --> New best: {best_loss:.4f} | model saved.", flush=True)

        save_checkpoint(CKPT_PATH, model, optimiser, scheduler, epoch, best_loss)
        print("Checkpoint saved.", flush=True)

    print("Training complete.", flush=True)
    wandb.finish()


if __name__ == "__main__":
    train()