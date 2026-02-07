import os
import shutil

dataset_path = "bone fracture detection.v4-v4.yolov8"

detect_path = "dataset_detection"
classify_path = "dataset_classification"

class_mapping = {
    0: 'elbow fracture',
    1: 'fingers fracture',
    2: 'forearm fracture',
    3: 'humerus fracture',
    4: 'humerus fracture',
    5: 'shoulder fracture',
    6: 'wrist fracture'
}

os.makedirs(detect_path, exist_ok=True)
os.makedirs(classify_path, exist_ok=True)

for c in ["normal", "fracture"]:
    os.makedirs(os.path.join(detect_path, c), exist_ok=True)

for c in set(class_mapping.values()):
    os.makedirs(os.path.join(classify_path, c), exist_ok=True)

image_exts = [".jpg", ".jpeg", ".png"]

for split in ['train','valid','test']:

    print(f"\nProcessing {split} dataset...")

    labels_dir = os.path.join(dataset_path, split, "labels")
    images_dir = os.path.join(dataset_path, split, "images")

    for label_file in os.listdir(labels_dir):

        label_path = os.path.join(labels_dir, label_file)
        base_name = label_file.replace(".txt","")

        # Find image
        image_path = None
        for ext in image_exts:
            temp = os.path.join(images_dir, base_name + ext)
            if os.path.exists(temp):
                image_path = temp
                break

        if image_path is None:
            continue

        with open(label_path) as f:
            line = f.readline().strip()

        # NORMAL IMAGE
        if not line:
            shutil.copy(image_path,
                os.path.join(detect_path,"normal",os.path.basename(image_path)))
            continue

        # FRACTURE IMAGE
        class_id = int(line.split()[0])

        shutil.copy(image_path,
            os.path.join(detect_path,"fracture",os.path.basename(image_path)))

        fracture_type = class_mapping[class_id]

        shutil.copy(image_path,
            os.path.join(classify_path,fracture_type,os.path.basename(image_path)))

print("\n✅ Dual dataset creation completed!")
