import os
import cv2
import numpy as np
import psutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


IMG_SIZE = 512
VALID_EXT = (".jpg", ".jpeg", ".png")
NUM_WORKERS = psutil.cpu_count(logical=False) - 1  # 13 physical cores, 1 left free


def crop_black_background(img, threshold=20, padding=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    h, w = img.shape[:2]
    k = max(5, w // 200)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(mask)
    x, y, w_box, h_box = cv2.boundingRect(coords)

    H, W = img.shape[:2]
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w_box + padding, W)
    y2 = min(y + h_box + padding, H)

    return img[y1:y2, x1:x2]


def process_single_image(args):
    in_file, out_file = args

    img = cv2.imread(in_file)
    img_cropped = crop_black_background(img)
    img_resized = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(out_file, img_resized)


def process_folder(input_path, output_path, desc=""):
    os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(input_path) if f.lower().endswith(VALID_EXT)]

    jobs = [
        (
            os.path.join(input_path, img_name),
            os.path.join(output_path, img_name)
        )
        for img_name in images
    ]

    with Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(
            pool.imap(process_single_image, jobs),
            total=len(jobs),
            desc=desc
        ))


def process_RFMid():
    INPUT_DIR = r"D:\Otago\3rdSemester\Research\Datasets\RFMid"
    OUTPUT_DIR = r"D:\Otago\3rdSemester\Research\Code\DATASETS_RESIZED\RFMid"

    splits = ["Test_set", "Training_set", "Validation_set"]

    for split in splits:
        input_path = os.path.join(INPUT_DIR, split)
        output_path = os.path.join(OUTPUT_DIR, split)
        process_folder(input_path, output_path, desc=f"RFMid {split}")


def process_ORIGIA():
    INPUT_DIR = r"D:\Otago\3rdSemester\Research\Datasets\ORIGIA"
    OUTPUT_DIR = r"D:\Otago\3rdSemester\Research\Code\DATASETS_RESIZED\ORIGIA"

    for split in ["train", "test"]:
        input_path = os.path.join(INPUT_DIR, split)
        output_path = os.path.join(OUTPUT_DIR, split)
        process_folder(input_path, output_path, desc=f"ORIGIA {split}")


def process_Eyepacs():
    INPUT_DIR = r"D:\Otago\3rdSemester\Research\Datasets\Eyepacs"
    OUTPUT_DIR = r"D:\Otago\3rdSemester\Research\Code\DATASETS_RESIZED"

    for split in ["train", "test", "val"]:
        for label in ["0", "1", "2", "3", "4"]:
            input_path = os.path.join(INPUT_DIR, split, label)
            output_path = os.path.join(OUTPUT_DIR, split, label)
            process_folder(input_path, output_path, desc=f"Eyepacs {split}/label {label}")


if __name__ == "__main__":
    print(f"Running with {NUM_WORKERS} workers...")
    process_Eyepacs()
    process_ORIGIA()
    process_RFMid()
    print("Done")