import os
import cv2
from tqdm import tqdm



IMG_SIZE = 512

VALID_EXT = (".jpg", ".jpeg", ".png")

def process_RFMid():
    INPUT_DIR = r"D:\Otago\3rdSemester\Research\Datasets\RFMid"
    OUTPUT_DIR = r"D:\Otago\3rdSemester\Research\Code\DATASETS_RESIZED\RFMid"

    splits = ["Test_set", "Training_set", "Validation_set"]

    for split in tqdm(splits, desc="RFMid Splits"):
        input_path = os.path.join(INPUT_DIR, split)
        output_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(output_path, exist_ok=True)

        images = [f for f in os.listdir(input_path) if f.lower().endswith(VALID_EXT)]
        for img_name in images:
            in_file = os.path.join(input_path, img_name)
            out_file = os.path.join(output_path, img_name)

            try:
                img = cv2.imread(in_file)
                if img is None:
                    continue

                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(out_file, img_resized)

            except Exception as e:
                print(f"Error processing {in_file}: {e}")


def process_ORIGIA():
    INPUT_DIR = r"D:\Otago\3rdSemester\Research\Datasets\ORIGIA"
    OUTPUT_DIR = r"D:\Otago\3rdSemester\Research\Code\DATASETS_RESIZED\ORIGIA"

    for split in tqdm(["train", "test"], desc="ORIGIA Splits"):
        input_path = os.path.join(INPUT_DIR, split)
        output_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(output_path, exist_ok=True)

        images = [f for f in os.listdir(input_path) if f.lower().endswith(VALID_EXT)]

        for img_name in images:
            in_file = os.path.join(input_path, img_name)
            out_file = os.path.join(output_path, img_name)

            try:
                img = cv2.imread(in_file)
                if img is None:
                    continue

                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(out_file, img_resized)
            except Exception as e:
                print(f"Error processing {in_file}: {e}")



def process_Eyepacs():
    INPUT_DIR = r"D:\Otago\3rdSemester\Research\Datasets\Eyepacs"
    OUTPUT_DIR = r"D:\Otago\3rdSemester\Research\Code\DATASETS_RESIZED"
    for split in tqdm(["train", "test", "val"], desc="Splits"):
        for label in tqdm(["0", "1", "2", "3", "4"], desc=f"{split} labels", leave=False):

            input_path = os.path.join(INPUT_DIR, split, label)
            output_path = os.path.join(OUTPUT_DIR, split, label)

            # create directory
            os.makedirs(output_path, exist_ok=True)

            images = [f for f in os.listdir(input_path) if f.lower().endswith(VALID_EXT)]

            for img_name in images:
                in_file = os.path.join(input_path, img_name)
                out_file = os.path.join(output_path, img_name)

                try:
                    img = cv2.imread(in_file)

                    if img is None:
                        continue

                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                    cv2.imwrite(out_file, img_resized)

                except Exception as e:
                    print(f"Error processing {in_file}: {e}")


if __name__ == "__main__":
    process_Eyepacs()
    process_ORIGIA()
    process_RFMid()
    print("Done")