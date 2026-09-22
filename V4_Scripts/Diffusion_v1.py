import os
import math
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from data_loader import get_combined_loader

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR     = "checkpoints/diffusion_5skips"
CKPT_PATH    = os.path.join(CKPT_DIR, "last.pt")
BEST_PATH    = os.path.join(CKPT_DIR, "best_diffusion_5skips.pt")
LR           = 1e-3
BATCH_SIZE   = 64
EPOCHS       = 50
T_MAX        = 1000   # total diffusion timesteps
TIME_EMB_DIM = 256
WANDB_KEY    = "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"


# ─── Noise schedule ───────────────────────────────────────────────────────────

class NoiseScheduler:
    """
    Linear beta schedule.
    Precomputes √ᾱₜ and √(1−ᾱₜ) for the forward process:
        xₜ = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        betas              = torch.linspace(beta_start, beta_end, T, device=device)
        alphas             = 1.0 - betas
        alpha_bar          = torch.cumprod(alphas, dim=0)

        self.T                        = T
        self.sqrt_alpha_bar           = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    def add_noise(self, x0, t, noise=None):
        """Returns (xₜ, ε) — noise is returned so it can be used as the training target."""
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sma = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return sa * x0 + sma * noise, noise


# ─── Timestep embedding ───────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for timestep t, followed by a two-layer MLP.
    Output shape: (B, time_emb_dim).
    """
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
        args = t[:, None].float() * freqs[None]           # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)
        return self.proj(emb)                              # (B, dim)


# ─── Building blocks ──────────────────────────────────────────────────────────

class DownBlock(nn.Module):
    """
    Strided conv down block — mirrors one encoder block from the CAE exactly:
        Conv2d(stride=2) → BatchNorm → LeakyReLU(0.1)
    Timestep embedding is injected as a channel-wise bias after activation.
    """
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
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x + t


class MiddleBlock(nn.Module):
    """
    Bottleneck block — no spatial change, two conv layers.
    This is the latent space (512×8×8) that gets extracted at inference.
    Timestep is injected before the convolutions.
    """
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
    Transposed conv up block — mirrors one decoder block from the CAE exactly:
        ConvTranspose2d(stride=2) → BatchNorm → LeakyReLU(0.1)
        → concat skip → Conv2d → BatchNorm → LeakyReLU(0.1)
    Timestep embedding is injected after upsampling, before the skip merge.
    """
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up    = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn_up  = nn.BatchNorm2d(out_ch)
        self.act_up = nn.LeakyReLU(0.1, inplace=True)
        self.merge  = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, skip, t_emb):
        x = self.act_up(self.bn_up(self.up(x)))
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t
        return self.merge(torch.cat([x, skip], dim=1))


# ─── Full U-Net ───────────────────────────────────────────────────────────────

class DiffusionUNet5Skips(nn.Module):
    """
    Diffusion U-Net with 5 skip connections.

    Channel progression matches the CAE encoder/decoder exactly:
        3×512×512  →  64×256×256  →  128×128×128  →  256×64×64
        →  512×32×32  →  512×16×16  →  512×8×8  (middle / latent)

    During training:
        forward(xₜ, t)  →  (ε̂, z)
        Loss = L1(ε̂, ε)

    During inference / embedding extraction:
        encode(x₀, t_fixed)  →  z  (512×8×8)
        z.mean(dim=(2,3))    →  512-dim embedding vector
    """

    def __init__(self, time_emb_dim=TIME_EMB_DIM):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # ── Down path: mirrors CAE Encoder blocks 1–6 ──
        self.down1 = DownBlock(3,   64,  time_emb_dim)  # → 64  × 256×256
        self.down2 = DownBlock(64,  128, time_emb_dim)  # → 128 × 128×128
        self.down3 = DownBlock(128, 256, time_emb_dim)  # → 256 × 64×64
        self.down4 = DownBlock(256, 512, time_emb_dim)  # → 512 × 32×32
        self.down5 = DownBlock(512, 512, time_emb_dim)  # → 512 × 16×16
        self.down6 = DownBlock(512, 512, time_emb_dim)  # → 512 × 8×8

        # ── Middle block — THIS is the latent z (512×8×8) ──
        self.middle = MiddleBlock(512, time_emb_dim)

        # ── Up path: 5 skip connections, mirrors CAE Decoder blocks 1–5 ──
        # up1: 512×8×8   → skip s5(512) → 512×16×16
        self.up1 = UpBlock(512, 512, 512, time_emb_dim)
        # up2: 512×16×16 → skip s4(512) → 512×32×32
        self.up2 = UpBlock(512, 512, 512, time_emb_dim)
        # up3: 512×32×32 → skip s3(256) → 256×64×64
        self.up3 = UpBlock(512, 256, 256, time_emb_dim)
        # up4: 256×64×64 → skip s2(128) → 128×128×128
        self.up4 = UpBlock(256, 128, 128, time_emb_dim)
        # up5: 128×128×128 → skip s1(64) → 64×256×256
        self.up5 = UpBlock(128, 64,  64,  time_emb_dim)

        # ── Final head: no skip, no BN — predicts noise ε̂ ──
        # Note: NO Sigmoid here. ε̂ is unbounded noise, not a pixel image.
        self.final_up = nn.ConvTranspose2d(
            64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x, t):
        """
        Args:
            x : noisy image xₜ, shape (B, 3, 512, 512)
            t : timestep indices,  shape (B,)
        Returns:
            eps_hat : predicted noise, shape (B, 3, 512, 512)
            z       : latent middle block, shape (B, 512, 8, 8)
        """
        t_emb = self.time_emb(t)         # (B, TIME_EMB_DIM)

        # Down — save skip connections
        s1 = self.down1(x,  t_emb)       # 64  × 256×256
        s2 = self.down2(s1, t_emb)       # 128 × 128×128
        s3 = self.down3(s2, t_emb)       # 256 × 64×64
        s4 = self.down4(s3, t_emb)       # 512 × 32×32
        s5 = self.down5(s4, t_emb)       # 512 × 16×16
        h  = self.down6(s5, t_emb)       # 512 × 8×8

        # Middle — latent
        z  = self.middle(h, t_emb)       # 512 × 8×8

        # Up with skips
        x  = self.up1(z,  s5, t_emb)    # 512 × 16×16
        x  = self.up2(x,  s4, t_emb)    # 512 × 32×32
        x  = self.up3(x,  s3, t_emb)    # 256 × 64×64
        x  = self.up4(x,  s2, t_emb)    # 128 × 128×128
        x  = self.up5(x,  s1, t_emb)    # 64  × 256×256

        eps_hat = self.final_up(x)       # 3   × 512×512
        return eps_hat, z

    @torch.no_grad()
    def encode(self, x, t):
        """
        Forward pass to bottleneck only — used for embedding extraction.
        Does NOT run the up path (faster, no gradient needed).

        Args:
            x : noisy image xₜ, shape (B, 3, 512, 512)
            t : timestep indices,  shape (B,)
        Returns:
            z : latent, shape (B, 512, 8, 8)
        """
        t_emb = self.time_emb(t)
        s1    = self.down1(x,  t_emb)
        s2    = self.down2(s1, t_emb)
        s3    = self.down3(s2, t_emb)
        s4    = self.down4(s3, t_emb)
        s5    = self.down5(s4, t_emb)
        h     = self.down6(s5, t_emb)
        return self.middle(h, t_emb)


criterion = nn.L1Loss()


# ─── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, noise_scheduler, t_fixed=250):
    """
    Extracts 512-dim embedding vectors for every image in loader.
    Uses a fixed timestep t_fixed (default 250) so embeddings are deterministic.
    Matches CAE's z.mean(dim=(2,3)) pooling convention.
    """
    model.eval()
    embeddings = []
    t_tensor   = torch.full((1,), t_fixed, device=DEVICE, dtype=torch.long)

    for imgs, _, _ in loader:
        imgs = imgs.to(DEVICE)
        B    = imgs.shape[0]
        t    = t_tensor.expand(B)
        xt, _= noise_scheduler.add_noise(imgs, t)
        z    = model.encode(xt, t)
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
    model           = DiffusionUNet5Skips().to(DEVICE)
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
        project="Diffusion-5skips",
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
            imgs = imgs.to(DEVICE)
            B    = imgs.shape[0]

            # Sample a random timestep per image in the batch
            t = torch.randint(0, T_MAX, (B,), device=DEVICE, dtype=torch.long)

            # Forward diffusion: x₀ → xₜ
            xt, noise = noise_scheduler.add_noise(imgs, t)

            optimiser.zero_grad()
            eps_hat, _ = model(xt, t)
            loss = criterion(eps_hat, noise)   # L1(ε̂, ε)
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
                imgs = imgs.to(DEVICE)
                B    = imgs.shape[0]
                t    = torch.randint(0, T_MAX, (B,), device=DEVICE, dtype=torch.long)
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

        # ── Save best model ────────────────────────────────────────────────────
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            with open(os.path.join(CKPT_DIR, "best_config.txt"), "w") as f:
                f.write(f"Best val loss: {best_loss:.6f}\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"lr: {LR}\n")
                f.write(f"batch_size: {BATCH_SIZE}\n")
            print(f"  --> New best: {best_loss:.4f} | model saved.", flush=True)

        # ── Save rolling checkpoint ────────────────────────────────────────────
        save_checkpoint(CKPT_PATH, model, optimiser, scheduler, epoch, best_loss)
        print("Checkpoint saved.", flush=True)

    print("Training complete.", flush=True)
    wandb.finish()


if __name__ == "__main__":
    train()