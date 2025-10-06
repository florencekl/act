import albumentations as A
from .gaussian_contrast import *
from .dropout import *

def clahe(p: float = 0.5):
    clahe = A.Sequential(
        [
            # A.FromFloat(max_value=255, dtype="uint8", always_apply=True),
            A.CLAHE(clip_limit=(4, 6), tile_grid_size=(8, 12), always_apply=True),
            # A.ToFloat(max_value=255, always_apply=True),
        ],
        p=p,
    )
    return clahe

intensity_transforms_names = [
    "clahe",
    "gaussian_blur",
    "motion_blur",
    "median_blur",
    "sharpen",
    "emboss",
    # "planckian_jitter",
    "random_brightness_contrast",
    "gaussian_contrast",
    "random_fog",
    # "random_snow",
    "random_rain",
    "dropout",
    "coarse_dropout",
    # "chromatic_aberration",
]
intensity_transforms_list = [
    clahe(),
    A.GaussianBlur((3, 5)),
    A.MotionBlur(blur_limit=(3, 5)),
    A.MedianBlur(blur_limit=5),
    A.Sharpen(alpha=(0.2, 0.5)),
    A.Emboss(alpha=(0.2, 0.5)),
    # A.PlanckianJitter(mode="blackbody"),
    A.RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-0.1, 0.1), ensure_safe_range=True),
    gaussian_contrast(alpha=(0.4, 2.0), sigma=(0.2, 1.0), max_value=1),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08),
    # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3),
    A.RandomRain(rain_type="drizzle", drop_width=1, blur_value=1),
    Dropout(dropout_prob=0.05),
    CoarseDropout(
        max_holes=20,  # More holes
        max_height=64,  # Larger holes
        max_width=64,
        min_holes=8,  # Minimum number of holes increased
        min_height=8,
        min_width=8,
        p=0.8,  # 
    ),
    # A.ChromaticAberration(),
]

def get_intensity_transforms(p: float = 1.0):
    intensity_transforms = A.SomeOf(
        [
            A.OneOf(
                [
                    A.GaussianBlur((3, 5)),
                    A.MotionBlur(blur_limit=(3, 5)),
                    A.MedianBlur(blur_limit=5),
                ],
            ),
            # A.OneOf(
            #     [
            #         A.Sharpen(alpha=(0.2, 0.5)),
            #         A.Emboss(alpha=(0.2, 0.5)),
            #     ],
            # ),
            # A.PlanckianJitter(mode="blackbody"),  # Fine

            # A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=(-0.2, 0.2)),
            # gaussian_contrast(alpha=(0.4, 2.0), sigma=(0.2, 1.0), max_value=1),
            # A.OneOf(
            #     [
            #         A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08),
            #         # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3),
            #         A.RandomRain(rain_type="drizzle", drop_width=1, blur_value=1),
            #     ],
            # ),
            A.OneOf(
                [
                    Dropout(dropout_prob=0.05),
                    CoarseDropout(
                        max_holes=12,
                        max_height=24,
                        max_width=24,
                        min_holes=4,
                        min_height=4,
                        min_width=4,
                        p=0.8,
                    ),
                ],
            ),
            # A.ChannelShuffle(),
            # A.ChromaticAberration(),
        ],
        n=np.random.randint(1, 5),
        replace=False,
        p=p,
        )
    return intensity_transforms

'''intensity_transforms = A.SomeOf(
    [
        A.OneOf(
            [
                A.GaussianBlur((3, 5)),
                A.MotionBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=5),
            ],
        ),
        A.OneOf(
            [
                A.Sharpen(alpha=(0.4, 0.7)),  # Stronger sharpening
                A.Emboss(alpha=(0.4, 0.7)),  # Stronger emboss effect
            ],
        ),
        A.PlanckianJitter(mode="blackbody"),  # Keeping this as it seems fine
        gaussian_contrast(alpha=(0.5, 1.5), sigma=(0.1, 0.5), max_value=1),  # Expanded range for contrast
        A.OneOf(
            [
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.08),  # Thicker fog
                A.RandomRain(rain_type="drizzle", drop_width=1, blur_value=1),  # Heavier rain with more blur
            ],
        ),
        A.OneOf(
            [
                Dropout(dropout_prob=0.1),  # Higher dropout probability
                CoarseDropout(
                    max_holes=12,
                    max_height=24,
                    max_width=24,
                    min_holes=4,
                    min_height=4,
                    min_width=4,
                    p=0.6,
                            ),
            ],
        ),
        A.ChromaticAberration()
    ],
    n=np.random.randint(2, 5),  # Increased minimum number of augmentations applied
    replace=False,
)'''