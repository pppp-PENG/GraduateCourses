# 1. Model in cifar10
### Structure
```python
Batch_size=1000

nn.Conv2d(3, 64, 7, stride=2, padding=3)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 128) - MyResBlock(128, 128) * 16
MyResBlock(128, 256) - MyResBlock(256, 256) * 8
MyResBlock(256, 512) - MyResBlock(512, 512) * 4
MyResBlock(512, 1024) - MyResBlock(1024, 1024) * 4
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(1024, 200)

parameters: 46.87M
```

### Preferance
```
[epoch=  1] loss: 4.7932
[epoch= 10] loss: 2.1081
[epoch= 64] loss: 0.1004
16s/epoch, 5it/s (RTX 4090)
Accuracy of the network on the 100000 train images: 99.00%
Accuracy of the network on the 10000 valid images: 38.82%
```

# 2. Model in cifar10
### Structure
```python
Batch_size=1000

nn.Conv2d(3, 64, 7, stride=2, padding=3)
nn.BatchNorm2d(64) - nn.ReLU(inplace=True)
MyResBlock(64, 256) - MyResBlock(256, 256) * 3
MyResBlock(256, 512) - MyResBlock(512, 512) * 6
MyResBlock(512, 1024) - MyResBlock(1024, 1024) * 12
nn.AdaptiveAvgPool2d((1,1)) - nn.Flatten() - nn.Linear(1024, 200)

parameters: 17.50M
```

### Preferance
```
[epoch=  1] loss: 5.3957
[epoch= 10] loss: 4.2945
[epoch= 50] loss: 2.8509
[epoch=128] loss: 1.6198
[epoch=256] loss: 0.9946
19s/epoch, 10it/s (RTX 4090)
Accuracy of the network on the 100000 train images: 76.44%
Accuracy of the network on the 10000 valid images: 46.40%
```