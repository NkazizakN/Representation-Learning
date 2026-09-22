import os
import math
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from data_loader import get_combined_loader

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR     = "checkpoints/diffusion_noskip"
CKPT_PATH    = os.path.join(CKPT_DIR, "last.pt")
BEST_PATH    = os.path.join(CKPT_DIR, "best_diffusion_noskip.pt")
LR           = 1e-3
BATCH_SIZE   = 64
EPOCHS       = 50
T_MAX        = 1000
TIME_EMB_DIM = 256
WANDB_KEY    = "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"


# ─── Noise schedule ───────────────────────────────────────────────────────────

class NoiseScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        betas     = torch.linspace(beta_start, beta_end, T, device=device)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)

        self.T                        = T
        self.sqrt_alpha_bar           = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sma = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sa * x0 + sma * noise, noise


# ─── Timestep embedding ───────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


# ─── Building blocks ──────────────────────────────────────────────────────────

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        x = self.conv(x)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return x + t


class MiddleBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.time_proj = nn.Linear(time_emb_dim, channels)

    def forward(self, x, t_emb):
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return self.conv(x + t)


class UpBlock(nn.Module):
    """
    Plain upsample — NO skip connection.
    ConvTranspose2d(stride=2) → BN → LeakyReLU(0.1) + timestep bias.
    All 5 up blocks use this. No concat, no merge conv.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up  = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        x = self.act(self.bn(self.up(x)))
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return x + t


# ─── Full U-Net ───────────────────────────────────────────────────────────────

class DiffusionUNetNoSkip(nn.Module):
    """
    Diffusion U-Net with no skip connections.

    Down path identical to 5skips and 2earlyskips — same 6 blocks,
    same channel progression, same 512×8×8 bottleneck.

    No skip connections at any level. All up blocks are plain UpBlock.

    Up path:
        up1: 512×8×8   → 512×16×16  — no skip
        up2: 512×16×16 → 512×32×32  — no skip
        up3: 512×32×32 → 256×64×64  — no skip
        up4: 256×64×64 → 128×128×128 — no skip
        up5: 128×128×128 → 64×256×256 — no skip
        final: 64×256×256 → 3×512×512 — no skip, no BN, no Sigmoid

    Latent extraction:
        model.encode(xt, t) → 512×8×8 middle block
        z.mean(dim=(2,3))   → 512-dim embedding vector
    """

    def __init__(self, time_emb_dim=TIME_EMB_DIM):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # ── Down path ──
        self.down1 = DownBlock(3,   64,  time_emb_dim)  # → 64  × 256×256
        self.down2 = DownBlock(64,  128, time_emb_dim)  # → 128 × 128×128
        self.down3 = DownBlock(128, 256, time_emb_dim)  # → 256 × 64×64
        self.down4 = DownBlock(256, 512, time_emb_dim)  # → 512 × 32×32
        self.down5 = DownBlock(512, 512, time_emb_dim)  # → 512 × 16×16
        self.down6 = DownBlock(512, 512, time_emb_dim)  # → 512 × 8×8

        # ── Middle block — latent z (512×8×8) ──
        self.middle = MiddleBlock(512, time_emb_dim)

        # ── Up path — all plain, no skips ──
        self.up1 = UpBlock(512, 512, time_emb_dim)  # 512×8×8   → 512×16×16
        self.up2 = UpBlock(512, 512, time_emb_dim)  # 512×16×16 → 512×32×32
        self.up3 = UpBlock(512, 256, time_emb_dim)  # 512×32×32 → 256×64×64
        self.up4 = UpBlock(256, 128, time_emb_dim)  # 256×64×64 → 128×128×128
        self.up5 = UpBlock(128, 64,  time_emb_dim)  # 128×128×128 → 64×256×256

        # ── Final head — no skip, no BN, no Sigmoid ──
        self.final_up = nn.ConvTranspose2d(
            64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x, t):
        """
        Args:
            x : noisy image xₜ  (B, 3, 512, 512)
            t : timestep indices (B,)
        Returns:
            eps_hat : predicted noise   (B, 3, 512, 512)
            z       : middle block latent (B, 512, 8, 8)
        """
        t_emb = self.time_emb(t)

        # Down — no skips saved
        x = self.down1(x,  t_emb)   # 64  × 256×256
        x = self.down2(x,  t_emb)   # 128 × 128×128
        x = self.down3(x,  t_emb)   # 256 × 64×64
        x = self.down4(x,  t_emb)   # 512 × 32×32
        x = self.down5(x,  t_emb)   # 512 × 16×16
        x = self.down6(x,  t_emb)   # 512 × 8×8

        # Middle — latent
        z = self.middle(x, t_emb)   # 512 × 8×8

        # Up — no skips
        x = self.up1(z,  t_emb)     # 512 × 16×16
        x = self.up2(x,  t_emb)     # 512 × 32×32
        x = self.up3(x,  t_emb)     # 256 × 64×64
        x = self.up4(x,  t_emb)     # 128 × 128×128
        x = self.up5(x,  t_emb)     # 64  × 256×256

        eps_hat = self.final_up(x)  # 3   × 512×512
        return eps_hat, z

    @torch.no_grad()
    def encode(self, x, t):
        """
        Down path + middle only — no up path.
        Used for embedding extraction at inference.
        """
        t_emb = self.time_emb(t)
        x     = self.down1(x,  t_emb)
        x     = self.down2(x,  t_emb)
        x     = self.down3(x,  t_emb)
        x     = self.down4(x,  t_emb)
        x     = self.down5(x,  t_emb)
        x     = self.down6(x,  t_emb)
        return self.middle(x, t_emb)


criterion = nn.L1Loss()


# ─── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, noise_scheduler, t_fixed=250):
    model.eval()
    embeddings = []
    t_tensor   = torch.full((1,), t_fixed, device=DEVICE, dtype=torch.long)

    for imgs, _, _ in loader:
        imgs  = imgs.to(DEVICE)
        B     = imgs.shape[0]
        t     = t_tensor.expand(B)
        xt, _ = noise_scheduler.add_noise(imgs, t)
        z     = model.encode(xt, t)
        embeddings.append(z.mean(dim=(2, 3)).cpu())

    return torch.cat(embeddings, dim=0)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

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


# ─── Training loop ────────────────────────────────────────────────────────────

def train():
    print("Loading dataloaders...", flush=True)
    train_loader = get_combined_loader(BATCH_SIZE, split="train")
    val_loader   = get_combined_loader(BATCH_SIZE, split="val")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    print("Building model...", flush=True)
    model           = DiffusionUNetNoSkip().to(DEVICE)
    noise_scheduler = NoiseScheduler(T=T_MAX, device=DEVICE)
    optimiser       = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler       = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        project="Diffusion-noskip",
        name=f"lr_{LR}_bs_{BATCH_SIZE}",
        config={
            "lr":           LR,
            "batch_size":   BATCH_SIZE,
            "epochs":       EPOCHS,
            "T_max":        T_MAX,
            "time_emb_dim": TIME_EMB_DIM,
        }
    )
    print("Wandb initialised.", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS} | lr={optimiser.param_groups[0]['lr']}", flush=True)

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True):
            imgs      = imgs.to(DEVICE)
            B         = imgs.shape[0]
            t         = torch.randint(0, T_MAX, (B,), device=DEVICE, dtype=torch.long)
            xt, noise = noise_scheduler.add_noise(imgs, t)

            optimiser.zero_grad()
            eps_hat, _ = model(xt, t)
            loss = criterion(eps_hat, noise)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train done | loss: {train_loss:.4f}", flush=True)

        # ── Validation ─────────────────────────────────────────────────────────
        print("Running validation...", flush=True)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, _, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=True):
                imgs      = imgs.to(DEVICE)
                B         = imgs.shape[0]
                t         = torch.randint(0, T_MAX, (B,), device=DEVICE, dtype=torch.long)
                xt, noise = noise_scheduler.add_noise(imgs, t)
                eps_hat, _ = model(xt, t)
                loss = criterion(eps_hat, noise)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        wandb.log({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         optimiser.param_groups[0]["lr"],
            "epoch":      epoch,
        })

        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}", flush=True)

        # ── Save best ──────────────────────────────────────────────────────────
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            with open(os.path.join(CKPT_DIR, "best_config.txt"), "w") as f:
                f.write(f"Best val loss: {best_loss:.6f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"lr: {LR}\n")
                f.write(f"batch_size: {BATCH_SIZE}\n")
            print(f"  --> New best: {best_loss:.4f} | model saved.", flush=True)

        # ── Rolling checkpoint ─────────────────────────────────────────────────
        save_checkpoint(CKPT_PATH, model, optimiser, scheduler, epoch, best_loss)
        print("Checkpoint saved.", flush=True)

    print("Training complete.", flush=True)
    wandb.finish()


if __name__ == "__main__":
    train()