import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import json
import copy
from torch.nn import functional as F


class PatchEmbedding(nn.Module):

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2


        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class SimpleViT(nn.Module):
    def __init__(self, *, image_size=224, patch_size=16, num_classes, dim, depth, heads, mlp_dim, in_channels=3):
        super().__init__()

        # 1. Patch Embedding 层
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 可学习的 [CLS] token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 位置编码

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation=F.gelu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        b = img.shape[0]

        # 图像分块与嵌入
        x = self.patch_embedding(img)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x += self.pos_embedding
        x = self.transformer_encoder(x)

        cls_token_output = x[:, 0]
        return self.mlp_head(cls_token_output)


def main():

    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    num_classes = len(cat_to_name)
    print(f"数据集包含 {num_classes} 个类别。")

    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"将在 {device} 上进行训练。")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                                             num_workers=4),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False,
                                             num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['valid']}")


    model = SimpleViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    # --- 保存模型 ---
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'num_classes': num_classes,
        'model_architecture': 'simple_vit_custom'
    }
    torch.save(checkpoint, 'flower_classifier_simple_vit_custom.pth')
    print("模型已保存为 flower_classifier_simple_vit_custom.pth")


if __name__ == '__main__':
    main()



#训练量小，导致模型过拟合
# 解决过拟合问题
