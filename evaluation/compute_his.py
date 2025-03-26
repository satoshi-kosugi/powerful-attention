import cv2
import numpy as np
import os, sys
import torch
from PIL import Image
from skimage import color
import torchvision.transforms as transforms
from tqdm import tqdm

device = "cuda"
grid_a = [torch.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(1, 21, 21, 512, 512).to(device),
            torch.linspace(-1, 1, 21).view(1, 21, 1, 1, 1).expand(1, 21, 21, 512, 512).to(device)]
grid_b = [torch.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(1, 21, 21, 512, 512).to(device),
            torch.linspace(-1, 1, 21).view(1, 1, 21, 1, 1).expand(1, 21, 21, 512, 512).to(device)]


def histogram_intersection(hist1, hist2):
    minima = np.minimum(hist1, hist2)
    intersection = np.sum(minima)
    return intersection

def calc_hist(path_imageA, path_imageB):
    hists = []

    for i in range(2):
        im_path = [path_imageA, path_imageB][i]
        im = Image.open(im_path).convert('RGB')
        im = np.array(im)
        if i == 0:
            im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
            im = cv2.resize(im, (512, 512))
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab).to(device)
        data_ab = lab_t[[1, 2], ...] / 110.0
        data_ab = data_ab.reshape((1, data_ab.shape[0], data_ab.shape[1], data_ab.shape[2]))

        N, C, H, W = data_ab.shape
        hist_a = torch.max(0.1 - torch.abs(grid_a[i] - data_ab[:, 0, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
        hist_b = torch.max(0.1 - torch.abs(grid_b[i] - data_ab[:, 1, :, :].view(N, 1, 1, H, W)), torch.Tensor([0]).to(device)) * 10
        hist = (hist_a * hist_b).mean(dim=(3, 4)).view(N, -1)
        hists.append(hist)

    hr = hists[0].data.cpu().float().numpy().flatten()
    hg = hists[1].data.cpu().float().numpy().flatten()
    intersect = cv2.compareHist(hr, hg, cv2.HISTCMP_INTERSECT)

    return intersect


def calculate_average_similarity(dirA, dirB):
    filesA = set(os.listdir(dirA))
    filesB = set(os.listdir(dirB))

    common_files = sorted(list(filesA.intersection(filesB)))

    total_similarity = 0
    count = 0

    for file_name in tqdm(common_files):
        path_imageA = os.path.join(dirA, file_name)
        path_imageB = os.path.join(dirB, file_name)

        similarity = calc_hist(path_imageA, path_imageB)
        total_similarity += similarity
        count += 1

    average_similarity = total_similarity / count
    print(f"HIS: {average_similarity:.4f}")

if __name__ == "__main__":
    dirA = sys.argv[1]
    dirB = sys.argv[2]

    print("Computing HIS...")
    calculate_average_similarity(dirA, dirB)
