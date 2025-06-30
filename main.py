import torch
import torch.nn as nn
from torchvision import transforms
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pystyle import Colors, Colorate

banner = '''

 █████╗ ██████╗ ████████╗██╗███████╗███████╗██╗  ██╗██████╗ ██████╗     
██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝██╔════╝╚██╗██╔╝╚════██╗██╔══██╗    
███████║██████╔╝   ██║   ██║█████╗  █████╗   ╚███╔╝  █████╔╝██║  ██║    
██╔══██║██╔══██╗   ██║   ██║██╔══╝  ██╔══╝   ██╔██╗ ██╔═══╝ ██║  ██║    
██║  ██║██║  ██║   ██║   ██║██║     ███████╗██╔╝ ██╗███████╗██████╔╝    
╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═════╝     
                                                                        '''

print(Colorate.Horizontal(Colors.red_to_yellow, (banner)))

class Simple2DGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Инициализация модели CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Кастомные слои генерации
        self.style_mapper = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        ).to(self.device)

    def generate_image(self, prompt, steps=100):
        """Генерация изображения по текстовому описанию"""
        print(f"Генерация: '{prompt}'...")
        
        # Кодирование текста
        text = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        
        # Генерация изображения
        style_code = self.style_mapper(text_features.float())
        img = self.image_decoder(style_code.view(-1, 128, 1, 1))
        
        # Оптимизация изображения
        img = self._optimize_image(img, text_features, steps)
        
        # Конвертация в PIL Image
        return self._tensor_to_image(img)

    def _optimize_image(self, img, text_features, steps):
        """Оптимизация изображения для лучшего соответствия тексту"""
        img = img.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([img], lr=0.05)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Вычисление потерь
            img_features = self.clip_model.encode_image(
                transforms.functional.resize(img, (224, 224)))
            loss = 1 - torch.cosine_similarity(text_features, img_features)
            
            # Регуляризация
            loss += 0.1 * (img[:, :, :-1, :] - img[:, :, 1:, :]).pow(2).mean()
            loss += 0.1 * (img[:, :, :, :-1] - img[:, :, :, 1:]).pow(2).mean()
            
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 20 == 0:
                print(f"Шаг {step + 1}/{steps}, Loss: {loss.item():.4f}")
        
        return img.detach()

    @staticmethod
    def _tensor_to_image(tensor):
        """Конвертация тензора в изображение"""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return Image.fromarray(img.astype('uint8'))

    @staticmethod
    def display_image(image, prompt):
        """Отображение изображения"""
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(prompt)
        plt.show()

# Пример использования
if __name__ == "__main__":
    # Инициализация генератора
    generator = Simple2DGenerator()
    
    # Промт задается как переменная
    prompt = "YOUR_PROMPT_HERE"
    
    # Генерация изображения
    image = generator.generate_image(prompt, steps=80)
    
    # Отображение результата
    generator.display_image(image, prompt)
