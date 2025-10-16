import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import copy
from typing import Optional, Tuple, Union


# 保留代码A中的ViT核心组件（与之前相同）
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


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 32,  # CIFAR-10是32×32
            patch_size: Union[int, Tuple[int, int]] = 8,  # 小图像用小patch
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
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 32/8=4 → 4×4=16个patch

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 {H}x{W} 与期望尺寸 {self.img_size} 不匹配"

        x = self.proj(x)  # 输出形状: B, embed_dim, 4, 4
        x = x.flatten(2).transpose(1, 2)  # 展平为 B, 16, embed_dim
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 应该能被 num_heads {num_heads} 整除"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask

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
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
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
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
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
            img_size: Union[int, Tuple[int, int]] = 32,  # 适配CIFAR-10的32×32
            patch_size: Union[int, Tuple[int, int]] = 8,
            in_chans: int = 3,
            num_classes: int = 10,  # CIFAR-10是10分类
            embed_dim: int = 256,
            depth: int = 10,
            num_heads: int = 4,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  # 16个patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1是加上cls_token
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self.head, 'weight'):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
        if hasattr(self.head, 'bias') and self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # B, 16, embed_dim

        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, embed_dim
        x = torch.cat((cls_tokens, x), dim=1)  # B, 17, embed_dim (16+1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, 0])  # 用cls_token的输出做分类

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# 针对CIFAR-10的模型配置
def create_cifar10_vit() -> VisionTransformer:
    """为CIFAR-10创建适配的ViT模型"""
    return VisionTransformer(
        img_size=32,  # CIFAR-10图像尺寸
        patch_size=8,  # 8×8的patch（32/8=4，生成4×4=16个patch，序列长度适中）
        embed_dim=256,  # 嵌入维度（比花卉模型稍大，因CIFAR数据更多）
        depth=10,  # 10层Transformer（适合中等数据量）
        num_heads=4,  # 4个注意力头（256/4=64，每个头维度合理）
        mlp_ratio=4.,  # MLP隐藏层维度为4×embed_dim
        qkv_bias=True,
        drop_rate=0.2,  # 适度dropout防止过拟合
        attn_drop_rate=0.1,
        drop_path_rate=0.1,  # 随机深度
        num_classes=10  # CIFAR-10是10分类
    )


def main():
    # 超参数设置（适配CIFAR-10）
    batch_size = 64  # CIFAR图像小，可加大batch
    num_epochs = 80  # 数据量更大，需要更多epoch
    learning_rate = 0.001  # 稍大于花卉模型（数据更多，可更快收敛）
    data_dir = './data/cifar10'  # 数据保存路径

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # CIFAR-10数据预处理（针对32×32图像优化）
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 边缘填充后随机裁剪（增强鲁棒性）
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            # CIFAR-10标准归一化参数
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]),
    }

    # 加载CIFAR-10数据集（无需手动整理文件夹，torchvision直接支持）
    image_datasets = {
        'train': datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=data_transforms['train']
        ),
        'test': datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=data_transforms['test']
        )
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4
        ),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4
        )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes  # CIFAR-10类别名（如airplane, car等）
    print(f"类别: {class_names}")
    print(f"训练集大小: {dataset_sizes['train']}, 测试集大小: {dataset_sizes['test']}")

    # 初始化模型
    model = create_cifar10_vit()
    model = model.to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    # 使用AdamW（带权重衰减的Adam，更适合Transformer）
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # 余弦退火学习率调度器（自动调整学习率，收敛更稳定）
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 训练过程
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch包含训练和测试阶段
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 训练模式（启用dropout等）
            else:
                model.eval()  # 评估模式（禁用dropout等）

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):  # 训练时计算梯度
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 训练阶段反向传播+参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 学习率调度（仅训练阶段）
            if phase == 'train':
                exp_lr_scheduler.step()

            # 计算每个epoch的损失和准确率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存测试集准确率最高的模型
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # 训练结束统计
    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳测试集准确率: {best_acc:4f}')

    # 保存最佳模型
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'cifar10_vit_best.pth')
    print("最佳模型已保存为 cifar10_vit_best.pth")


if __name__ == '__main__':
    main()