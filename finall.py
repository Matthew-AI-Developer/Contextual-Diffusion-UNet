import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_rate=0.1, num_groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels) 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.gelu(self.norm1(x)))
        
        time_emb = self.time_mlp(t)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb 

        h = self.conv2(self.gelu(self.norm2(h)))
        h = self.dropout(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels, n_feat=64, n_cfeat=10, n_classes=10, num_groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.n_classes = n_classes
        self.num_groups = num_groups

        self.time_embed = SinusoidalEmbedding(n_feat)

        self.initial_conv = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(num_groups, n_feat) 
        self.initial_gelu = nn.GELU()

        self.down1_res = ResidualBlock(n_feat, n_feat, n_feat, num_groups=num_groups)
        self.down1_pool = nn.MaxPool2d(2)

        self.down2_res = ResidualBlock(n_feat, 2 * n_feat, n_feat, num_groups=num_groups)
        self.down2_pool = nn.MaxPool2d(2)
        
        self.mid_block1 = ResidualBlock(2 * n_feat, 2 * n_feat, n_feat, num_groups=num_groups)
        self.mid_block2 = ResidualBlock(2 * n_feat, 2 * n_feat, n_feat, num_groups=num_groups) 

        self.contextembed = nn.Sequential(
            nn.Linear(n_classes, n_cfeat),
            nn.GELU(),
            nn.Linear(n_cfeat, n_feat)
        )
        
        self.up_conv1 = nn.ConvTranspose2d(2 * n_feat, n_feat, 2, 2)
        self.up_norm1 = nn.GroupNorm(num_groups, n_feat)
        self.up_gelu1 = nn.GELU()
        self.up_res_block1 = ResidualBlock(n_feat + n_feat, n_feat, n_feat, num_groups=num_groups)

        self.up_conv2 = nn.ConvTranspose2d(n_feat, n_feat, 2, 2)
        self.up_norm2 = nn.GroupNorm(num_groups, n_feat)
        self.up_gelu2 = nn.GELU()
        self.up_res_block2 = ResidualBlock(n_feat + n_feat, n_feat, n_feat, num_groups=num_groups)
        
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, in_channels, 3, 1, 1), 
        )

    def forward(self, x, c, t, context_mask):
        context_emb = self.contextembed(c)
        context_emb = torch.where(context_mask.view(-1, 1), context_emb, torch.zeros_like(context_emb))
        time_emb = self.time_embed(t.float() * 1000)
        combined_emb = time_emb + context_emb

        x_init = self.initial_gelu(self.initial_norm(self.initial_conv(x))) 
        
        # Downsampling
        down1_out = self.down1_res(x_init, combined_emb) 
        down1_out = self.down1_pool(down1_out) 

        down2_out = self.down2_res(down1_out, combined_emb) 
        down2_out = self.down2_pool(down2_out) 
        
        # Bottleneck
        h = self.mid_block1(down2_out, combined_emb)
        h = self.mid_block2(h, combined_emb) 

        h = self.up_conv1(h) 
        h = self.up_gelu1(self.up_norm1(h))
        h = torch.cat((down1_out, h), dim=1) 
        h = self.up_res_block1(h, combined_emb) 

        h = self.up_conv2(h) 
        h = self.up_gelu2(self.up_norm2(h))
        h = torch.cat((x_init, h), dim=1) 
        h = self.up_res_block2(h, combined_emb) 

        output = self.out(h) 
        return output

class DiffusionModel:
    def __init__(self, model, n_steps=1000, min_beta=1e-4, max_beta=0.02, device='cuda', beta_schedule='linear'):
        super().__init__()
        self.n_steps = n_steps
        self.device = device

        if beta_schedule == 'linear':
            self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        elif beta_schedule == 'cosine':
            timesteps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps + 0.008
            alphas_cumprod = torch.cos(((timesteps / (timesteps[-1])) * math.pi / 2)).pow(2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0, 0.999).to(device)
        else:
            raise ValueError("Choose linear or cosine")
            
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = model

    def forward_process(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1. - alpha_bar_t) * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, n_samples, size, device, guide_w=2.0):
        x_i = torch.randn(n_samples, *size).to(device)
        
        c_i = torch.arange(0, 10).to(device)
        if n_samples > c_i.shape[0]:
            c_i = c_i.repeat(n_samples // c_i.shape[0] + 1)[:n_samples]
        
        c_i_onehot = F.one_hot(c_i, num_classes=10).float().to(device)

        for i in tqdm(range(self.n_steps - 1, -1, -1), desc="Sampling"):
            t_is = torch.tensor([i]).to(device)
            t_is = t_is.repeat(n_samples)

            z = torch.randn(n_samples, *size).to(device) if i > 0 else 0

            context_mask_cond = torch.ones_like(c_i_onehot[:, 0]).bool().to(device)
            eps_cond = self.model(x_i, c_i_onehot, t_is, context_mask_cond) 
            
            context_mask_uncond = torch.zeros_like(c_i_onehot[:, 0]).bool().to(device)
            eps_uncond = self.model(x_i, c_i_onehot, t_is, context_mask_uncond)
            
            eps = (1 + guide_w) * eps_cond - guide_w * eps_uncond
            
            sqrt_alpha_t = torch.sqrt(self.alphas[i])
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bars[i])

            x_i = (
                (1 / sqrt_alpha_t) *
                (x_i - (beta_t / sqrt_one_minus_alpha_bar_t) * eps) +
                torch.sqrt(beta_t) * z
            )
        return x_i

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    n_epoch = 5
    batch_size = 128
    n_feat = 64
    lrate = 1e-4
    
    model = UNet(in_channels=1, n_feat=n_feat, n_classes=10).to(device)
    diffusion = DiffusionModel(model, device=device, beta_schedule='cosine') 
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for ep in range(n_epoch):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{n_epoch}")
        for x, c in pbar:
            optimizer.zero_grad()
            x = x.to(device)
            c = F.one_hot(c, num_classes=10).float().to(device)
            
            t = torch.randint(0, diffusion.n_steps, (x.shape[0],)).to(device)
            
            uncond_prob = 0.1 
            context_mask = torch.rand(x.shape[0], device=device) > uncond_prob
            
            xt, noise = diffusion.forward_process(x, t)
            
            predicted_noise = model(xt, c, t, context_mask.view(-1, 1)) 
            
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"diffusion_model_epoch_{ep+1}.pth")
        
        torch.manual_seed(42) 
        generated_images = diffusion.sample(10, (1, 28, 28), device)
        generated_images = (generated_images + 1) / 2
        from torchvision.utils import save_image
        save_image(generated_images, f"generated_images_epoch_{ep+1}.png", nrow=5)
        print(f"Generated images saved for epoch {ep+1} ")


if __name__ == "__main__":
    train()