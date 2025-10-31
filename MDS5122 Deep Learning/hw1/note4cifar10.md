# 1. Baseline
### Structure
```python
Conv2d(3, 128, 3, padding=1) - ReLU(inplace=True) - MaxPool2d(2) - Dropout(0.3)
Conv2d(128, 256, 3, padding=1) - ReLU(inplace=True) - MaxPool2d(2) - Dropout(0.3)
Conv2d(256, 512, 3, padding=1) - ReLU(inplace=True)
Conv2d(512, 512, 3, padding=1) - ReLU(inplace=True)
Conv2d(512, 256, 3, padding=1) - ReLU(inplace=True) - MaxPool2d(2) - Dropout(0.3)
Flatten()
Linear(256 * 4 * 4, 512) - ReLU(inplace=True) - Dropout(0.5)
Linear(512, 256) - ReLU(inplace=True) - Dropout(0.5)
Linear(256, 128) - ReLU(inplace=True) - Dropout(0.5)
Linear(128, 10)

parameters: 7.28M
```

### Preferance
```
[epoch=  1] loss: 1.966
[epoch= 10] loss: 0.902
[epoch= 50] loss: 0.389
[epoch=128] loss: 0.227
25s/epoch, 30it/s (T4)
Accuracy: 87.17%
```

# 2. Residual Mechanism
### Structure
```python
Conv2d(3, 128, 3, padding=1) - ReLU(inplace=True) - MaxPool2d(2) - Dropout(0.3)
Conv2d(128, 256, 3, padding=1) - ReLU(inplace=True) - MaxPool2d(2) - Dropout(0.3)
MyResBlock() * 3
MaxPool2d(2) - ReLU(inplace=True) - Dropout(0.3)
Flatten()
Linear(256 * 4 * 4, 512) - ReLU(inplace=True) - Dropout(0.5)
Linear(512, 256) - ReLU(inplace=True) - Dropout(0.5)
Linear(256, 128) - ReLU(inplace=True) - Dropout(0.5)
Linear(128, 10)

parameters: 6.10M
```

### Preferance
```
[epoch=  1] loss: 1.922
[epoch= 10] loss: 0.913
[epoch= 50] loss: 0.460
[epoch=128] loss: 0.321
5s/epoch, 155it/s (RTX 4090)
Accuracy: 87.43%
```

# 3. Deeper & Wider
### Structure
```python
nn.Conv2d(3, 64, 3, padding=1)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 128) - MyResBlock(128, 128)
MyResBlock(128, 256) - MyResBlock(256, 256)
MyResBlock(256, 512) - MyResBlock(512, 512)
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(512, 10)

parameters: 11.03M
```

### Preferance
```
[epoch=  1] loss: 1.467
[epoch= 10] loss: 0.412
[epoch= 50] loss: 0.094
[epoch=128] loss: 0.051
6s/epoch, 115it/s (RTX 4090)
Accuracy: 91.09%
```

# 4. Optimizer Improve
### Optimizer
```
AdamW: lr=0.001
optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.5)
```

### Structure
```python
nn.Conv2d(3, 64, 3, padding=1)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 128) - MyResBlock(128, 128)
MyResBlock(128, 256) - MyResBlock(256, 256)
MyResBlock(256, 512) - MyResBlock(512, 512)
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(512, 10)

parameters: 11.03M
```

### Preferance
```
[epoch=  1] loss: 1.460
[epoch= 10] loss: 0.412
[epoch= 50] loss: 0.057
[epoch=128] loss: 0.008
6s/epoch, 115it/s (RTX 4090)
Accuracy: 92.57%
```

# 5. Data Augmentation
### Data
```
tv_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
```

### Structure
```python
nn.Conv2d(3, 64, 3, padding=1)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 128) - MyResBlock(128, 128)
MyResBlock(128, 256) - MyResBlock(256, 256)
MyResBlock(256, 512) - MyResBlock(512, 512)
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(512, 10)

parameters: 11.03M
```

### Preferance
```
[epoch=  1] loss: 1.591
[epoch= 10] loss: 0.448
[epoch= 50] loss: 0.067
[epoch=128] loss: 0.010
6s/epoch, 115it/s (RTX 4090)
Accuracy: 92.61%
```



# 6. Attention
### Structure
```python
nn.Conv2d(3, 64, 3, padding=1)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 128) - MyAttentionBlock(128, 128)
MyResBlock(128, 256) - MyAttentionBlock(256, 256)
MyResBlock(256, 512) - MyAttentionBlock(512, 512)
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(512, 10)

parameters: 8.97M
```

### Preferance
```
[epoch=  1] loss: 1.418
[epoch= 10] loss: 0.491
[epoch= 50] loss: 0.084
[epoch=128] loss: 0.015
10s/epoch, 75it/s (RTX 4090)
Accuracy: 90.85%
```

# 7. Other Changes
### Optimizer
```
AdamW: lr=0.00003
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### Structure
```python
Batch_size=256

nn.Conv2d(3, 64, 3, padding=1)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 64)
MyResBlock(64, 128) - MyResBlock(128, 128) * 2 - nn.Dropout(0.05)
MyResBlock(128, 256) - MyResBlock(256, 256) * 4 - nn.Dropout(0.05)
MyResBlock(256, 512) - MyResBlock(512, 512) * 2 - nn.Dropout(0.05)
MyResBlock(512, 1024) - MyResBlock(1024, 1024) * 1 - nn.Dropout(0.05)
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(1024, 10)

parameters: 53.24M
```

### Preferance
```
[epoch=  1] loss: 1.567
[epoch= 10] loss: 0.442
[epoch= 50] loss: 0.072
[epoch=128] loss: 0.004
10s/epoch, 19it/s (RTX 4090)
Accuracy: 93.24%
```
