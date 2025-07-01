from io import BytesIO

import requests
import torch
import webdataset as wds
from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

URLS = [
    "https://huggingface.co/datasets/laion/laion400m-data/resolve/main/laion400m-data/laion400m-part-000001.tar",
    "https://huggingface.co/datasets/laion/laion400m-data/resolve/main/laion400m-data/laion400m-part-000002.tar",
]


class StreamingImageDataset(IterableDataset):
    def __init__(
        self,
        urls,
        transform=None,
        max_samples=None,
    ):
        self.urls = urls
        self.transform = transform or transforms.ToTensor()
        self.max_samples = max_samples

    def __iter__(self):
        pipeline = wds.WebDataset(self.urls)
                    .decode("pil")
                    .to_tuple("jpg", "txt")
