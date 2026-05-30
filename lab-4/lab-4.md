```
%pip install -q facenet-pytorch torchmetrics torch-fidelity
```

```
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from tqdm.notebook import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
```

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Работаем на: {device}")

source_dir = '/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba'
attr_file = '/kaggle/input/datasets/jessicali9530/celeba-dataset/list_attr_celeba.csv'

cropped_dir = '/kaggle/working/cropped_faces'
os.makedirs(cropped_dir, exist_ok=True)

if not os.path.exists(source_dir) or not os.path.exists(attr_file):
    print("ERROR: Kaggle не видит датасет")
else:
    print(f"Датасет найден! В исходной папке {len(os.listdir(source_dir))} файлов.")
    print(f"Папка для вырезанных лиц готова: {cropped_dir}")

os.makedirs(cropped_dir, exist_ok=True)
mtcnn = MTCNN(image_size=64, margin=10, keep_all=False, post_process=False, device=device)

df_attr = pd.read_csv(attr_file)
df_attr['Male'] = df_attr['Male'].replace(-1, 0)
attr_dict = dict(zip(df_attr['image_id'], df_attr['Male']))

NUM_IMAGES_TO_PROCESS = 10000

already_cropped = len(os.listdir(cropped_dir))
if already_cropped < 100:
    print("Начинаем детектировать и вырезать лица...")
    
    all_imgs = [f for f in sorted(os.listdir(source_dir)) if f.endswith('.jpg')][:NUM_IMAGES_TO_PROCESS]
    
    if len(all_imgs) == 0:
        raise ValueError("В папке source_dir нет картинок .jpg! Проверь пути.")

    for img_name in tqdm(all_imgs):
        try:
            img_path = os.path.join(source_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            face = mtcnn(img, save_path=os.path.join(cropped_dir, img_name))
        except Exception as e:
            pass

print(f"Лица вырезаны! :))) Сейчас в папке {cropped_dir} лежит {len(os.listdir(cropped_dir))} картинок.")
```

```
class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_dict):
        self.root_dir = root_dir
        self.img_names = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.attr_dict = attr_dict
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.attr_dict.get(img_name, 0)
        return image, torch.tensor(label, dtype=torch.long)

dataset = CelebADataset(cropped_dir, attr_dict)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
print(f"Всего изображений для обучения: {len(dataset)}")
```

```
class Critic(nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        self.conditional = conditional
        self.label_emb = nn.Embedding(2, 64*64) if conditional else None
        
        in_channels = 4 if conditional else 3
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def forward(self, img, labels=None):
        if self.conditional and labels is not None:
            c = self.label_emb(labels).view(-1, 1, 64, 64)
            img = torch.cat([img, c], dim=1)
        return self.model(img).view(-1)

class Generator(nn.Module):
    def __init__(self, z_dim=100, conditional=False):
        super().__init__()
        self.conditional = conditional
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(2, 10) if conditional else None
        
        in_channels = z_dim + 10 if conditional else z_dim
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels=None):
        if self.conditional and labels is not None:
            c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
            noise = torch.cat([noise, c], dim=1)
        return self.model(noise)

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, conditional):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates, labels)
    fake = torch.ones((real_samples.size(0),), requires_grad=False).to(device)
    
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

```
def train_wgan_gp(generator, critic, dataloader, epochs=5, conditional=False):
    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))
    
    history_G, history_C = [], []
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    z_dim = 100

    print(f"Начинаем обучение ({'conditional' if conditional else 'unconditional'})")
    for epoch in range(epochs):
        loss_g_epoch, loss_c_epoch = 0, 0
        for batch_idx, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch+1}/{epochs}")):
            imgs, labels = imgs.to(device), labels.to(device)
            cur_batch_size = imgs.shape[0]

            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
                fake_imgs = generator(noise, labels if conditional else None)
                
                critic_real = critic(imgs, labels if conditional else None)
                critic_fake = critic(fake_imgs.detach(), labels if conditional else None)
                
                gp = compute_gradient_penalty(critic, imgs.data, fake_imgs.data, labels if conditional else None, conditional)
                
                # функция потерь Вассерштейна + штраф
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_C.step()

            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake_imgs = generator(noise, labels if conditional else None)
            gen_fake = critic(fake_imgs, labels if conditional else None)
            
            loss_gen = -torch.mean(gen_fake)
            
            generator.zero_grad()
            loss_gen.backward()
            opt_G.step()
            
            loss_c_epoch += loss_critic.item()
            loss_g_epoch += loss_gen.item()

        history_C.append(loss_c_epoch / len(dataloader))
        history_G.append(loss_g_epoch / len(dataloader))
    
    return history_G, history_C
```

```
gen_uncond = Generator(conditional=False).to(device)
crit_uncond = Critic(conditional=False).to(device)
hist_G_unc, hist_C_unc = train_wgan_gp(gen_uncond, crit_uncond, dataloader, epochs=15, conditional=False)

gen_cond = Generator(conditional=True).to(device)
crit_cond = Critic(conditional=True).to(device)
hist_G_cond, hist_C_cond = train_wgan_gp(gen_cond, crit_cond, dataloader, epochs=15, conditional=True)
```

```
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_G_unc, label='Generator Loss')
plt.plot(hist_C_unc, label='Critic Loss')
plt.title('Безусловная WGAN-GP')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_G_cond, label='Generator Loss')
plt.plot(hist_C_cond, label='Critic Loss')
plt.title('Условная WGAN-GP (Пол)')
plt.legend()
plt.show()
```

```
def to_uint8(img_tensor):
    return ((img_tensor + 1) / 2 * 255).to(torch.uint8)

print("Считаем FID и IS")
fid = FrechetInceptionDistance(feature=64).to(device)
inception = InceptionScore().to(device)

gen_cond.eval()
with torch.no_grad():
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        fid.update(to_uint8(real_imgs), real=True)
        break
        
    # фейки
    noise = torch.randn(128, 100, 1, 1).to(device)
    random_labels = torch.randint(0, 2, (128,)).to(device)
    fake_imgs = gen_cond(noise, random_labels)
    
    fid.update(to_uint8(fake_imgs), real=False)
    inception.update(to_uint8(fake_imgs))

print(f"FID Score: {fid.compute().item():.2f} (less is better)")
is_mean, is_std = inception.compute()
print(f"Inception Score (IS): {is_mean.item():.2f} ± {is_std.item():.2f} (more is better)")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Сверху: женщины, снизу: мужчины")

with torch.no_grad():
    # женщины
    noise = torch.randn(5, 100, 1, 1).to(device)
    labels_w = torch.zeros(5, dtype=torch.long).to(device)
    fake_w = gen_cond(noise, labels_w).cpu()
    
    # мужчины
    noise = torch.randn(5, 100, 1, 1).to(device)
    labels_m = torch.ones(5, dtype=torch.long).to(device)
    fake_m = gen_cond(noise, labels_m).cpu()

for i in range(5):
    img_w = ((fake_w[i] + 1) / 2).permute(1, 2, 0).numpy()
    img_m = ((fake_m[i] + 1) / 2).permute(1, 2, 0).numpy()
    
    axes[0, i].imshow(img_w)
    axes[0, i].axis('off')
    axes[1, i].imshow(img_m)
    axes[1, i].axis('off')

plt.show()
```