import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import time
import os
import json
import copy
from PIL import Image


# =================================================================================
# --- 核心修改：自定义 Dataset，将数据预加载到内存 ---
# =================================================================================
class InMemoryDataset(Dataset):
    """
    一个将 ImageFolder 数据集完全加载到内存中的包装类，以消除 I/O 瓶颈。
    """

    def __init__(self, root, transform=None):
        print(f"开始将数据集从 {root} 预加载到内存...")
        start_time = time.time()

        # 使用 ImageFolder 来方便地获取所有图片的路径和标签
        original_dataset = datasets.ImageFolder(root)

        self.samples = []
        self.targets = []
        self.transform = transform
        self.class_to_idx = original_dataset.class_to_idx

        # 遍历 ImageFolder 中的每一个样本
        for path, target in original_dataset.samples:
            # 打开图片文件并保持为 PIL.Image 对象
            with Image.open(path).convert('RGB') as img:
                self.samples.append(img.copy())  # .copy() 确保文件句柄被释放
            self.targets.append(target)

        end_time = time.time()
        print(f"加载完成！耗时 {end_time - start_time:.2f} 秒。共加载 {len(self.samples)} 张图片。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 从内存中直接获取 PIL Image 对象和标签
        sample = self.samples[index]
        target = self.targets[index]

        # 应用数据增强转换
        if self.transform:
            sample = self.transform(sample)

        return sample, target


def main():
    """
    主训练函数。
    """
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    num_classes = len(cat_to_name)
    print(f"数据集包含 {num_classes} 个类别。")

    batch_size = 64
    num_epochs = 25
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

    # 使用我们新的内存数据集类
    image_datasets = {
        'train': InMemoryDataset(train_dir, data_transforms['train']),
        'valid': InMemoryDataset(valid_dir, data_transforms['valid'])
    }

    # 确定合适的 num_workers
    # 对于内存数据集，I/O 瓶颈已消除，主要工作是数据增强。num_workers 依然有帮助。
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"使用 {num_workers} 个 worker 进程进行数据增强。")

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             persistent_workers=True if num_workers > 0 else False
                                             ),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             persistent_workers=True if num_workers > 0 else False
                                             )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['valid']}")

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.heads.head.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 训练循环部分保持不变 ---
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
            # 使用 tqdm 可以更好地观察数据加载速度
            # from tqdm import tqdm
            # for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
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
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'num_classes': num_classes,
        'model_architecture': 'vit_b_16'
    }
    torch.save(checkpoint, 'flower_classifier_vit_b_16.pth')
    print("模型已保存为 flower_classifier_vit_b_16.pth")


if __name__ == '__main__':
    main()

