import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from utils.dataloader import CelebADataset
from models.generator import Generator
from models.discriminator import Discriminator

# =============== CONFIGURATION ===============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "datasets/celeba"
IMAGE_SIZE = 64
LATENT_DIM = 100
STYLE_DIM = 512
BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 50
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =============== DATA LOADER ===============
dataset = CelebADataset(DATASET_PATH, IMAGE_SIZE)
pin_memory = True if DEVICE == "cuda" else False
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=pin_memory
)

# =============== MODELS ===============
generator = Generator(latent_dim=LATENT_DIM, style_dim=STYLE_DIM).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# =============== OPTIMIZERS ===============
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.99))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.99))

# =============== LOSS FUNCTION ===============
adversarial_loss = nn.BCEWithLogitsLoss()

def gradient_penalty(real_imgs, fake_imgs):
    """Implements WGAN-GP gradient penalty for stabilizing GAN training."""
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=DEVICE)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(batch_size, 1, device=DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

if __name__ == "__main__":
    # =============== RESUME LOGIC ===============
    best_g_loss = float("inf")
    start_epoch = 1

    last_ckpt = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("generator_epoch")],
        reverse=True
    )
    if last_ckpt:
        latest_epoch = int(last_ckpt[0].split("_epoch")[1].split(".")[0])
        gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch{latest_epoch}.pth")
        disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{latest_epoch}.pth")

        print(f"[INFO] Loading checkpoints from epoch {latest_epoch}")
        generator.load_state_dict(torch.load(gen_path, map_location=DEVICE))
        discriminator.load_state_dict(torch.load(disc_path, map_location=DEVICE))

        start_epoch = latest_epoch + 1
        print(f"[INFO] Resuming training from epoch {start_epoch}")

    # Create fixed noise for evaluation
    fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)

    # =============== TRAINING LOOP ===============
    for epoch in range(start_epoch, EPOCHS + 1):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS}")
        save_interval = max(len(dataloader) // 10, 1)  # save 10 images per epoch

        for batch_idx, real_imgs in pbar:
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # ======= Train Discriminator =======
            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(noise).detach()
            real_logits = discriminator(real_imgs)
            fake_logits = discriminator(fake_imgs)
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            d_loss_real = adversarial_loss(real_logits, real_labels)
            d_loss_fake = adversarial_loss(fake_logits, fake_labels)
            gp = gradient_penalty(real_imgs, fake_imgs)
            d_loss = d_loss_real + d_loss_fake + 10 * gp
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # ======= Train Generator =======
            noise = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(noise)
            fake_logits = discriminator(gen_imgs)
            g_loss = adversarial_loss(fake_logits, real_labels)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            pbar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}"
            })

            # ======= Save progress images =======
            if (batch_idx + 1) % save_interval == 0:
                vutils.save_image(
                    gen_imgs[:1],
                    os.path.join(LOGS_DIR, f"epoch{epoch}_batch{batch_idx+1}.png"),
                    normalize=True, value_range=(-1, 1)
                )

        # ======= Save evaluation image =======
        generator.eval()
        with torch.no_grad():
            fixed_imgs = generator(fixed_noise)
        vutils.save_image(
            fixed_imgs,
            os.path.join(LOGS_DIR, f"epoch{epoch}_evaluation.png"),
            nrow=4,
            normalize=True, value_range=(-1, 1)
        )
        generator.train()

        # ======= Save best model =======
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_best.pth"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_best.pth"))

        # ======= Save latest epoch checkpoints =======
        torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth"))
