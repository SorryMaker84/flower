import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import json
import copy
from typing import Optional, Tuple, Union
import numpy as np


# 改进的DropPath实现（增加随机性）
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False,
              scale_by_keep: bool = True) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# 改进的PatchEmbed（增加位置编码多样性）
class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 256,
            bias: bool = True,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias
        )
        # 增加可学习的位置偏置
        self.pos_bias = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 {H}x{W} 与期望尺寸 {self.img_size} 不匹配"

        x = self.proj(x)  # B, embed_dim, grid_h, grid_w
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        x = x + self.pos_bias  # 增加位置偏置
        return x


# 改进的Attention（增加局部注意力机制）
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            local_window: int = 7  # 局部注意力窗口大小
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 应该能被 num_heads {num_heads} 整除"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.local_window = local_window

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # 生成局部注意力掩码
        if self.local_window > 0 and N > 1:  # 排除cls_token的情况
            local_mask = torch.ones((N, N), device=x.device) * -float('inf')
            for i in range(N):
                start = max(0, i - self.local_window)
                end = min(N, i + self.local_window + 1)
                local_mask[i, start:end] = 0
            if attn_mask is None:
                attn_mask = local_mask
            else:
                attn_mask = attn_mask + local_mask

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 每个形状为 (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)  # 适配注意力头维度

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 2  # 增大隐藏层比例

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        # 增加中间层提升非线性表达能力
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop)
        self.fc3 = nn.Linear(hidden_features // 2, out_features)
        self.drop3 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.drop3(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            local_window: int = 7
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
            local_window=local_window  # 传入局部窗口参数
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 102,
            embed_dim: int = 256,
            depth: int = 10,
            num_heads: int = 4,
            mlp_ratio: float = 3.5,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            act_layer: nn.Module = nn.GELU,
            local_window: int = 7
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 增加可学习的类别token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减（线性增加）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                local_window=local_window
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        # 增加分类头的表达能力
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            act_layer(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, num_classes)
        ) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 对卷积层和线性层使用更合适的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # B, num_patches, embed_dim

        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, embed_dim
        x = torch.cat((cls_tokens, x), dim=1)  # B, num_patches+1, embed_dim
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def create_flower_vit(num_classes: int = 102) -> VisionTransformer:
    """优化的花卉识别ViT模型（不使用预训练）"""
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=256,  # 增大嵌入维度提升特征表达
        depth=10,  # 增加层数增强特征提取
        num_heads=4,  # 4头注意力（256/4=64维度/头）
        mlp_ratio=3.5,
        qkv_bias=True,
        drop_rate=0.4,  # 增强dropout抑制过拟合
        attn_drop_rate=0.3,
        drop_path_rate=0.2,  # 提高随机深度比例
        num_classes=num_classes,
        local_window=5  # 局部注意力窗口
    )


def main():
    # 数据路径设置
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    num_classes = len(cat_to_name)
    print(f"数据集包含 {num_classes} 个类别。")

    # 超参数设置（针对小数据集优化）
    batch_size = 32
    num_epochs = 150  # 增加训练轮次
    base_lr = 3e-4  # 基础学习率
    weight_decay = 2e-4  # 权重衰减

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"将在 {device} 上进行训练。")

    # 增强数据预处理（更强烈的数据增强）
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),  # 提前转换为Tensor
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # 提前转换为Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 数据加载
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])
    }

    # 计算类别权重（处理类别不平衡）
    train_counts = np.bincount(image_datasets['train'].targets)
    class_weights = torch.FloatTensor(
        [len(image_datasets['train']) / (num_classes * count) for count in train_counts]
    ).to(device)

    # 使用加权采样解决类别不平衡
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=[class_weights[label].item() for label in image_datasets['train'].targets],
        num_samples=len(image_datasets['train']),
        replacement=True
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            sampler=sampler,  # 使用加权采样器
            num_workers=4
        ),
        'valid': torch.utils.data.DataLoader(
            image_datasets['valid'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['valid']}")

    # 创建模型
    model = create_flower_vit(num_classes=num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 使用类别权重
    optimizer = optim.AdamW(  # 使用AdamW优化器
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度器（带预热的余弦退火）
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 预热阶段线性增长
        else:
            # 余弦退火阶段
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 训练过程（增加早停机制）
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 20  # 早停耐心值
    counter = 0  # 早停计数器

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
                        # 梯度裁剪防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型并检查早停
            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0  # 重置计数器
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"早停于第{epoch + 1}轮")
                        break  # 跳出验证阶段循环

        # 检查是否需要早停
        if phase == 'valid' and counter >= patience:
            break  # 跳出epoch循环

        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:4f}')

    # 保存最佳模型
    model.load_state_dict(best_model_wts)
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'num_classes': num_classes,
        'model_architecture': 'vision_transformer'
    }
    torch.save(checkpoint, 'flower_classifier_vit_optimized.pth')
    print("优化模型已保存为 flower_classifier_vit_optimized.pth")


if __name__ == '__main__':
    main()