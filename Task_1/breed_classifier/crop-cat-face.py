import cv2
import dlib
import os
from glob import glob

DATA_DIR = 'pycatfd/data'
DETECTOR_SVM = os.path.join(DATA_DIR, 'detector.svm')
PREDICTOR_DAT = os.path.join(DATA_DIR, 'predictor.dat')
INPUT_IMAGE_DIR = 'cat breeds/images'
OUTPUT_DIR = 'cat_faces'
LOG_FILE = 'no_face_detected.txt'

detector = dlib.simple_object_detector(DETECTOR_SVM)

breed_dirs = [d for d in glob(os.path.join(INPUT_IMAGE_DIR, '*')) if os.path.isdir(d)]

with open(LOG_FILE, 'w') as log:
    log.write("Images with no face detected:\n")

cropped_images = []
for breed_dir in breed_dirs:
    breed_name = os.path.basename(breed_dir)
    output_breed_dir = os.path.join(OUTPUT_DIR, breed_name)
    os.makedirs(output_breed_dir, exist_ok=True)
    
    image_paths = glob(os.path.join(breed_dir, '*.jpg'))
    
    print(f"Processing breed: {breed_name} ({len(image_paths)} images)")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            with open(LOG_FILE, 'a') as log:
                log.write(f"{img_path} (failed to load)\n")
            continue
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        if height < 224 or width < 224:
            scale = max(224 / height, 224 / width)
            new_width = int((width + 1) * scale)
            new_height = int((height + 1) * scale)

            # Resize the image
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Detect faces with upsampling
        dets = detector(img_rgb, 1)  
        if len(dets) == 0:
            print(f"No face detected in: {img_path}")
            with open(LOG_FILE, 'a') as log:
                log.write(f"{img_path}\n")
            continue
        det = dets[0]
        
        # Original crop coordinates without padding
        x1 = max(det.left(), 0)
        y1 = max(det.top(), 0)
        x2 = min(det.right(), img.shape[1])
        y2 = min(det.bottom(), img.shape[0])

        w = x2 - x1
        h = y2 - y1


        if w > 224 or h > 224:
            cropped_face = img[y1:y2, x1:x2]
            cropped_face = cv2.resize(cropped_face, (224, 224))
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            new_x1 = cx - 112
            new_x2 = cx + 112
            new_y1 = cy - 112
            new_y2 = cy + 112

            if(new_x1 < 0):
                new_x2 = new_x2 - new_x1
                new_x1 = 0
            if(new_y1 < 0):
                new_y2 = new_y2 - new_y1
                new_y1 = 0
            if(new_x2 > img.shape[1]):
                new_x1 = new_x1 - (new_x2 - img.shape[1])
                new_x2 = img.shape[1]
            if(new_y2 > img.shape[0]):
                new_y1 = new_y1 - (new_y2 - img.shape[0])
                new_y2 = img.shape[0]

            cropped_face = img[new_y1:new_y2, new_x1:new_x2]

         
        
        output_filename = os.path.join(output_breed_dir, os.path.basename(img_path))
        cv2.imwrite(output_filename, cropped_face)
        print(f"Saved cropped face: {output_filename}")
        
        if len(cropped_images) < 5:  
            cropped_images.append((f"{breed_name}/{os.path.basename(img_path)}", cropped_face))



with open(LOG_FILE, 'r') as log:
    failed_count = len(log.readlines()) - 1
print(f"Number of images with no face detected: {failed_count}")

