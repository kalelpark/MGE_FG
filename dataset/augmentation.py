import os
import imp
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

train_transform_v2 = transforms.Compose([
                      transforms.Resize(512),
                      transforms.RandomCrop(448),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize,
                      ])

test_transform = transforms.Compose([
                     transforms.Resize(512),
                     transforms.CenterCrop(448),
                     transforms.ToTensor(),
                     normalize
                     ])
