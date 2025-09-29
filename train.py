import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import json
import copy


def main():

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

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # 冻结预训练模型的参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类头
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.heads.head.parameters(), lr=learning_rate)

    # 定义学习率调度器
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

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

            # 如果在验证集上得到更好的准确率，则保存模型权重
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
    # 在 checkpoint 中记录模型架构
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'num_classes': num_classes,
        'model_architecture': 'vit_b_16' # 修改架构名称
    }

    torch.save(checkpoint, 'flower_classifier_vit_b_16.pth')
    print("模型已保存为 flower_classifier_vit_b_16.pth")


if __name__ == '__main__':
    main()
