import torch
import torchvision.transforms as tv_transforms


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_type):
        self.dataset = dataset
        if data_type == "train":
            self.transformation = tv_transforms.Compose(
                [
                    tv_transforms.RandomRotation(degrees=15),
                    tv_transforms.RandomHorizontalFlip(),
                    tv_transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transformation = tv_transforms.Compose(
                [
                    tv_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {
            "image": self.transformation(self.dataset[idx]["image"]),
            "label": self.dataset[idx]["label"],
        }
        return item
