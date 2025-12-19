import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# =========================
# PARAMETRI
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 32
BATCH_SIZE = 32          # meglio 32 su CPU
EPOCHS = 50
LATENT_DIM = 128
LR_G = 1e-4
LR_D = 1e-4

N_CRITIC = 3           # quante volte alleni il Critic per 1 step di G
LAMBDA_GP = 10.0         # gradient penalty weight

SEED = 42
torch.manual_seed(SEED)

# =========================
# CARTELLE
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "gan_data")

OUTPUT_DIR = "wgan_gp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# TRASFORMAZIONI
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),  # augmentation utile con pochi dati
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
print("Classi trovate:", dataset.classes)
print("Numero immagini:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,          # prova 0/2/4 in base al PC
    pin_memory=(DEVICE == "cuda"),
    drop_last=True          # importante per GP (batch costante)
)

# =========================
# UTILS
# =========================
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# =========================
# GENERATOR (64x64)
# =========================
class Generator(nn.Module):
    def __init__(self, z_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),   # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),    # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),      # 32x32
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# =========================
# CRITIC (NO Sigmoid!)
# =========================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),   # 16x16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 8x8
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),# 4x4
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0))


# =========================
# GRADIENT PENALTY
# =========================
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    scores = critic(x_hat)  # (N,)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (N, C, H, W)

    grad = grad.view(batch_size, -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# =========================
# INIT
# =========================
G = Generator().to(DEVICE)
D = Critic().to(DEVICE)
G.apply(weights_init)
D.apply(weights_init)

opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)

# =========================
# TRAIN LOOP
# =========================
print("DEVICE:", DEVICE)
global_step = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    for i, (real, _) in enumerate(loader, start=1):
        real = real.to(DEVICE)

        # ============
        # Train Critic
        # ============
        for _ in range(N_CRITIC):
            z = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=DEVICE)
            fake = G(z).detach()

            d_real = D(real).mean()
            d_fake = D(fake).mean()
            gp = gradient_penalty(D, real, fake, DEVICE)

            loss_D = (d_fake - d_real) + LAMBDA_GP * gp

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

        # ============
        # Train Generator
        # ============
        z = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=DEVICE)
        fake = G(z)
        loss_G = -D(fake).mean()

        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

        global_step += 1

    dt = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f} | {dt:.1f}s")

    # Salva samples ogni 5 epoche
    if epoch % 5 == 0 or epoch == 1:
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, os.path.join(OUTPUT_DIR, f"samples_epoch_{epoch}.png"), normalize=True)

        # checkpoint
        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
        }, os.path.join(OUTPUT_DIR, f"ckpt_epoch_{epoch}.pt"))

print("Done. Output in:", OUTPUT_DIR)
