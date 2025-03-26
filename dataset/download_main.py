import requests
from bs4 import BeautifulSoup
import os
import shutil
from tqdm import tqdm
from PIL import Image

input_dir = "dataset/main/input"
reference_dir = "dataset/main/reference"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)

def save_image(img_url, save_path):
    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
    del response

def resize_image(image_path, size=(512, 512)):
    image = Image.open(image_path)
    image = image.resize(size)
    image.save(image_path)

def download_images_from_page(url, gray_pattern, ref_pattern, cnt_folder, sty_folder, start_num):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    images = soup.find_all('img')
    cnt_num = start_num

    for img in tqdm(images):
        src = img.get('src')
        if src and gray_pattern in src:
            gray_img_url = url.rsplit('/', 1)[0] + '/' + src
            ref_img_url = gray_img_url.replace(gray_pattern, ref_pattern).replace("_in.JPEG", "_ref.JPEG").replace("_input.JPEG", "_top_ref.JPEG")

            gray_save_path = os.path.join(cnt_folder, f"{cnt_num:03d}.png")
            ref_save_path = os.path.join(sty_folder, f"{cnt_num:03d}.png")

            save_image(gray_img_url, gray_save_path)
            save_image(ref_img_url, ref_save_path)

            resize_image(gray_save_path)
            resize_image(ref_save_path)

            cnt_num += 1

    return cnt_num

urls_and_patterns = [
    ("https://www.dongdongchen.bid/supp/deep_exam_colorization/index.html",
     "images/example_based/Input/gray_", "images/example_based/Reference/ref_"),
    ("https://www.dongdongchen.bid/supp/deep_exam_colorization/learning_based.html",
     "images/learning_based/in/", "images/learning_based/ref/"),
    ("https://www.dongdongchen.bid/supp/deep_exam_colorization/legacy_photos.html",
     "images/legacy/gray/in_", "images/legacy/ref/ref_"),
    ("https://www.dongdongchen.bid/supp/deep_exam_colorization/user_study.html",
     "images/user_study/input/", "images/user_study/our_top_ref/")
]

current_image_num = 1

for url, gray_pattern, ref_pattern in urls_and_patterns:
    print(url)
    current_image_num = download_images_from_page(url, gray_pattern, ref_pattern,
                                                  input_dir, reference_dir,
                                                  current_image_num)
