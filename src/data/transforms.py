import cv2
import albumentations as albu
from albumentations import pytorch as AT

from params import MEAN, STD


def color_transforms(p):
    return albu.OneOf(
        [
            albu.RandomGamma(always_apply=True),
            albu.RandomBrightnessContrast(always_apply=True),
        ],
        p=p,
    )


def distortion_transforms(p):
    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=1,
                sigma=5,
                alpha_affine=10,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        ],
        p=p,
    )

def normalizer(mean=MEAN, std=STD):
    return albu.Compose([
                albu.Normalize(mean=mean, std=std),
                AT.transforms.ToTensor(),
            ],
            p=1,
        )


def get_transfos(augment=True, visualize=False):
    if visualize:
        return albu.Compose([
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=45, p=0.5),
                color_transforms(p=0.5),
                distortion_transforms(p=0.5),
            ])

    if augment:
        return albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=45, p=0.5),
                color_transforms(p=0.5),
                distortion_transforms(p=0.5),
                normalizer(),
            ]
        )
    else:
        return normalizer()
