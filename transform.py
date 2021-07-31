import torchvision
from imgaug import augmenters as iaa
import numpy as np
import PIL

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.SomeOf((1, 2), [
            iaa.GammaContrast((0.5, 2.0)),
            iaa.Multiply(),
            iaa.GaussianBlur(1.0),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)  # 增加飽和度
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size=32):
        s = 0.5
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                #ImgAugTransform(),
                #lambda x: PIL.Image.fromarray(x),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                )
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                #torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                )
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
