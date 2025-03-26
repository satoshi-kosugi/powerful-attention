import os
import urllib.request
from PIL import Image
import cv2
from tqdm import tqdm

input_dir = "dataset/historical/input"
reference_dir = "dataset/historical/reference"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)

with open("dataset/historical.txt", "r") as file:
    lines = file.readlines()

for i, line in tqdm(enumerate(lines), total=len(lines)):
    file_number = f"{i+1:03}"

    A_url, B_url = line.strip().split(", ")

    A_temp = "dataset/tempA.jpg"
    urllib.request.urlretrieve(A_url, A_temp)

    img_A = Image.open(A_temp).convert("L")
    img_A = img_A.resize((512, 512))
    A_save_path = os.path.join(input_dir, f"{file_number}.png")
    img_A.save(A_save_path)

    B_temp = "dataset/tempB.jpg"
    urllib.request.urlretrieve(B_url, B_temp)

    img_B = Image.open(B_temp)
    img_B = img_B.resize((512, 512))
    B_save_path = os.path.join(reference_dir, f"{file_number}.png")
    img_B.save(B_save_path)

    os.remove(A_temp)
    os.remove(B_temp)
