import numpy as np
import os
import random
import logging
import pandas as pd 
from typing import Tuple, Optional, Dict, List, Any
from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
from tkinter import ttk, messagebox
from collections import OrderedDict 




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


NUM_LABEL_SETS = 3

EXAMPLE_DATA_ROOT = "Dataset/train" 

CSV_FILE_PATH = "Dataset/submit/predictions_filled_modified_0.csv"

TEST_IMAGE_BASE_DIR = "Dataset" 

NUM_TESTS = 100 
NUM_EXAMPLE_IMAGES = 7 
GUI_IMAGE_DISPLAY_SIZE = 150 
ALLOWED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff') 
EXAMPLE_COLS = 4 
LABEL_PADDING_LENGTH = 6 

CSV_FILENAME_COL = 'filename' 

CSV_LABEL_COLS = [f'label_{i+1}' for i in range(NUM_LABEL_SETS)] 

def load_data_from_csv(csv_path: str, num_label_sets: int, filename_col: str, label_cols: List[str], base_dir: Optional[str] = None, pad_length: int = 0) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    logging.info(f"--- Loading Test Data from CSV: {csv_path} ---")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return None, None

    try:
        
        df = pd.read_csv(csv_path, dtype=str)
        logging.info(f"CSV loaded successfully with {len(df)} rows.")
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        return None, None

    
    required_cols = [filename_col] + label_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns in CSV: {missing_cols}")
        return None, None

    
    
    df[filename_col] = df[filename_col].fillna('').astype(str)
    test_image_filenames = df[filename_col].tolist()

    
    ground_truth_labels_list = df[label_cols].fillna('').astype(str).values.tolist()

    
    valid_indices = [idx for idx, fname in enumerate(test_image_filenames) if fname]
    if len(valid_indices) < len(test_image_filenames):
        logging.warning(f"Removed {len(test_image_filenames) - len(valid_indices)} rows from CSV due to missing filenames.")
        test_image_filenames = [test_image_filenames[i] for i in valid_indices]
        ground_truth_labels_list = [ground_truth_labels_list[i] for i in valid_indices]

    if not test_image_filenames:
        logging.error("No valid filenames found in the CSV file after filtering.")
        return None, None

    
    test_image_paths = []
    if base_dir and os.path.isdir(base_dir):
        logging.info(f"Prepending base directory '{base_dir}' to test image filenames from CSV.")
        test_image_paths = [os.path.join(base_dir, fname) for fname in test_image_filenames]
    elif base_dir:
        logging.warning(f"Specified test image base directory '{base_dir}' not found or not a directory. Assuming CSV contains full paths.")
        test_image_paths = test_image_filenames
    else:
        logging.info("No test image base directory specified. Assuming CSV contains full paths.")
        test_image_paths = test_image_filenames

    
    for i, gt_list in enumerate(ground_truth_labels_list):
         if len(gt_list) != num_label_sets:
             logging.error(f"Data error in CSV row (approx index {i}): Found {len(gt_list)} labels, expected {num_label_sets} based on CSV_LABEL_COLS.")
             messagebox.showerror("CSV Error", f"Incorrect number of labels found in CSV row (approx index {i+1}). Expected {num_label_sets} based on config. Check CSV file.")
             return None, None 

    
    if pad_length > 0:
        logging.info(f"Padding labels to length {pad_length} with leading zeros.")
        padded_labels_list = []
        for label_set in ground_truth_labels_list:
            padded_set = []
            for label in label_set:
                if label: 
                    try:
                        
                        padded_set.append(str(label).zfill(pad_length))
                    except Exception as e:
                        logging.warning(f"Could not pad label '{label}': {e}. Keeping original.")
                        padded_set.append(label) 
                else:
                    padded_set.append('') 
            padded_labels_list.append(padded_set)
        ground_truth_labels_list = padded_labels_list 
        logging.debug(f"Example padded labels (first row): {ground_truth_labels_list[0] if ground_truth_labels_list else 'N/A'}")


    logging.info(f"Successfully processed CSV. Found {len(test_image_paths)} test images with valid filenames and {num_label_sets} labels each.")
    return test_image_paths, ground_truth_labels_list



def get_all_example_image_paths(base_example_dir: str) -> Dict[str, List[str]]:
    
    label_to_paths: Dict[str, List[str]] = {}
    logging.info(f"--- Scanning for ALL example images in base directory: {base_example_dir} ---")

    if not os.path.isdir(base_example_dir):
        logging.error(f"Example image base directory not found: {base_example_dir}")
        return {} 

    try:
        
        label_dirs = [d for d in os.listdir(base_example_dir) if os.path.isdir(os.path.join(base_example_dir, d))]
    except OSError as e:
        logging.error(f"Could not list directories in {base_example_dir}: {e}")
        return {}

    if not label_dirs:
        logging.warning(f"No subdirectories (labels for examples) found in {base_example_dir}")
        return {}

    logging.info(f"Scanning {len(label_dirs)} potential label directories in {base_example_dir} for example images.")
    total_example_images_found = 0

    for label_name in label_dirs:
        class_dir = os.path.join(base_example_dir, label_name)
        label_to_paths[label_name] = [] 
        images_in_label = 0
        try:
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                if os.path.isfile(img_path) and img_name.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                    label_to_paths[label_name].append(img_path)
                    images_in_label += 1
        except OSError as e:
            logging.warning(f"Could not read directory {class_dir}: {e}")
            continue 

        if images_in_label > 0:
            total_example_images_found += images_in_label
            logging.debug(f"Found {images_in_label} images for label '{label_name}'")
        else:
            
            logging.debug(f"No valid image files found in example directory: {class_dir}")
            

    logging.info(f"Found a total of {total_example_images_found} example images across all {len(label_dirs)} scanned label directories in {base_example_dir}.")
    if total_example_images_found == 0 and label_dirs:
         logging.warning(f"No example image files found within any subdirectories of {base_example_dir}")

    return label_to_paths





class SimpleImageVerifierGUI:
    def __init__(self, root: tk.Tk,
                 
                 example_label_paths_map: Dict[str, List[str]],
                 
                 test_image_paths_list: List[str],
                 ground_truth_labels_list: List[List[str]], 
                 num_label_sets: int,
                 num_tests_requested: int):
        self.root = root
        
        self.example_label_paths_map = example_label_paths_map
        self.num_label_sets = num_label_sets

        
        self.all_test_image_paths = test_image_paths_list
        self.all_ground_truth_labels = ground_truth_labels_list 

        
        self.current_test_index = 0 
        self.selected_data_indices: List[int] = [] 
        self.num_tests = 0 
        self.pass_count = 0 
        self.fail_count = 0 

        
        self.current_gt_labels: List[str] = []


        
        if not self.all_test_image_paths:
            logging.error("No test images loaded from CSV. Exiting GUI setup.")
            messagebox.showerror("Error", f"No valid image paths found or loaded from the CSV ({CSV_FILE_PATH}). Cannot start.")
            root.quit()
            return
        if len(self.all_test_image_paths) != len(self.all_ground_truth_labels):
             logging.error(f"Mismatch between number of test paths ({len(self.all_test_image_paths)}) and ground truth labels ({len(self.all_ground_truth_labels)}) from CSV.")
             messagebox.showerror("Error", "Inconsistent data loaded from CSV (path/label count mismatch). Check logs.")
             root.quit()
             return
        

        
        if not self.example_label_paths_map:
             
             logging.warning(f"The example label-to-paths map is empty (scanned from {EXAMPLE_DATA_ROOT}). No examples will be shown.")
             

        
        num_available = len(self.all_test_image_paths)
        actual_num_tests = min(num_tests_requested, num_available)
        if actual_num_tests < num_tests_requested:
            logging.warning(f"Requested {num_tests_requested} tests, but only {num_available} test images available in CSV {CSV_FILE_PATH}. Testing {actual_num_tests}.")
        if actual_num_tests == 0:
            logging.error("Zero test images available from CSV for verification.")
            messagebox.showerror("Error", f"Zero usable images found in {CSV_FILE_PATH} for testing.")
            root.quit()
            return
        self.num_tests = actual_num_tests
        
        self.selected_data_indices = random.sample(range(num_available), self.num_tests)


        
        self.root.title(f"Simple Image Verifier (0/{self.num_tests})")
        self.root.minsize(800, 650) 
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0) 
        main_frame.rowconfigure(1, weight=0) 
        main_frame.rowconfigure(2, weight=1) 
        main_frame.rowconfigure(3, weight=0) 

        
        test_frame = ttk.LabelFrame(main_frame, text="Test Image (from CSV)", padding="10")
        test_frame.grid(row=0, column=0, pady=5, padx=10, sticky="ew")
        test_frame.columnconfigure(0, weight=1)
        self.test_image_label = ttk.Label(test_frame, text="Loading test image...")
        self.test_image_label.grid(row=0, column=0, pady=5)
        self.test_image_path_label = ttk.Label(test_frame, text="", wraplength=750)
        self.test_image_path_label.grid(row=1, column=0, pady=2)

        
        gt_frame = ttk.LabelFrame(main_frame, text="Labels from CSV (Folder Names)", padding="10")
        gt_frame.grid(row=1, column=0, pady=5, padx=10, sticky="ew")
        gt_frame.columnconfigure(0, weight=1)
        self.ground_truth_labels_display: List[ttk.Label] = []
        for i in range(self.num_label_sets):
            
            label_col_name = CSV_LABEL_COLS[i] if i < len(CSV_LABEL_COLS) else f"Label Set {i+1}"
            label = ttk.Label(gt_frame, text=f"{label_col_name}: -", anchor="w")
            label.grid(row=i, column=0, sticky="ew", pady=1)
            self.ground_truth_labels_display.append(label)

        
        example_notebook_frame = ttk.LabelFrame(main_frame, text="Example Images for Labels from CSV", padding="10")
        example_notebook_frame.grid(row=2, column=0, pady=5, padx=10, sticky="nsew")
        example_notebook_frame.rowconfigure(0, weight=1)
        example_notebook_frame.columnconfigure(0, weight=1)

        self.example_notebook = ttk.Notebook(example_notebook_frame)
        self.example_notebook.grid(row=0, column=0, sticky="nsew")

        self.model_example_frames: List[ttk.Frame] = [] 
        for i in range(self.num_label_sets):
            tab_content_frame = ttk.Frame(self.example_notebook, padding="5")
            tab_content_frame.grid(sticky="nsew")
            
            for c in range(EXAMPLE_COLS):
                 tab_content_frame.columnconfigure(c, weight=1)
            
            tab_content_frame.rowconfigure(0, weight=0)

            
            canvas = tk.Canvas(tab_content_frame)
            scrollbar = ttk.Scrollbar(tab_content_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas) 

            scrollable_frame.bind(
                "<Configure>",
                lambda e, c=canvas: c.configure(scrollregion=c.bbox("all"))
            )
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            
            canvas.grid(row=1, column=0, sticky="nsew", columnspan=EXAMPLE_COLS) 
            scrollbar.grid(row=1, column=EXAMPLE_COLS, sticky="ns") 
            tab_content_frame.rowconfigure(1, weight=1) 

            
            tab_text = CSV_LABEL_COLS[i] if i < len(CSV_LABEL_COLS) else f"Label Set {i+1}"
            self.example_notebook.add(tab_content_frame, text=f"{tab_text} Examples")
            self.model_example_frames.append(scrollable_frame) 


        
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=3, column=0, pady=10, sticky="ew")
        button_frame.columnconfigure(0, weight=1) 
        button_frame.columnconfigure(1, weight=0) 
        button_frame.columnconfigure(2, weight=0) 
        button_frame.columnconfigure(3, weight=1) 
        self.pass_button = ttk.Button(button_frame, text="Pass (Looks Correct)", command=self.record_pass, width=20)
        self.pass_button.grid(row=0, column=1, padx=10)
        self.fail_button = ttk.Button(button_frame, text="Fail (Looks Incorrect)", command=self.record_fail, width=20)
        self.fail_button.grid(row=0, column=2, padx=10)

        
        self.load_next_test()

    def _load_and_display_image(self, img_path: str, label_widget: ttk.Label, display_size: int):
        
        
        error_img = Image.new('RGB', (display_size, int(display_size * 0.6)), color = 'lightgrey')
        try:
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 10), "Error", fill="red")
        except ImportError:
            pass 
        placeholder_tk = ImageTk.PhotoImage(error_img)


        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

            img = Image.open(img_path).convert('RGB')
            
            resampling_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            img.thumbnail((display_size, display_size), resampling_filter)
            img_tk = ImageTk.PhotoImage(img)
            label_widget.config(image=img_tk, text="") 
            label_widget.image = img_tk 
        except FileNotFoundError:
            label_widget.config(image=placeholder_tk, text=f"Not Found:\n{os.path.basename(img_path)}")
            label_widget.image = placeholder_tk
            logging.warning(f"Image file not found: {img_path}")
        except UnidentifiedImageError:
            label_widget.config(image=placeholder_tk, text=f"Cannot Read:\n{os.path.basename(img_path)}")
            label_widget.image = placeholder_tk
            logging.warning(f"Could not read image file (unidentified format): {img_path}")
        except Exception as e:
            label_widget.config(image=placeholder_tk, text=f"Load Error:\n{os.path.basename(img_path)}")
            label_widget.image = placeholder_tk
            logging.error(f"Error loading image {img_path} for GUI: {e}")

    def _clear_specific_example_frame(self, label_set_index: int):
        
        if 0 <= label_set_index < len(self.model_example_frames):
            frame_to_clear = self.model_example_frames[label_set_index]
            for widget in frame_to_clear.winfo_children():
                widget.destroy()
        else:
            logging.error(f"Attempted to clear invalid example frame index: {label_set_index}")


    def load_next_test(self):
        
        if self.current_test_index >= self.num_tests:
            self.show_final_results()
            return

        self.root.title(f"Simple Image Verifier ({self.current_test_index + 1}/{self.num_tests})")

        
        current_data_idx = self.selected_data_indices[self.current_test_index]

        
        current_img_path = self.all_test_image_paths[current_data_idx]
        self.current_gt_labels = self.all_ground_truth_labels[current_data_idx] 

        
        display_path = current_img_path
        if len(display_path) > 100: 
            display_path = "..." + display_path[-97:]
        self.test_image_path_label.config(text=f"Path: {display_path}")
        self._load_and_display_image(current_img_path, self.test_image_label, display_size=GUI_IMAGE_DISPLAY_SIZE * 2)

        
        for i in range(self.num_label_sets):
             label_col_name = CSV_LABEL_COLS[i] if i < len(CSV_LABEL_COLS) else f"Label Set {i+1}"
             
             gt_text = self.current_gt_labels[i] if self.current_gt_labels[i] else "N/A" 
             self.ground_truth_labels_display[i].config(text=f"{label_col_name}: {gt_text}")

        
        for i in range(self.num_label_sets):
            self._clear_specific_example_frame(i) 

            target_scrollable_frame = self.model_example_frames[i]
            
            current_label_name = self.current_gt_labels[i]
            

            if not current_label_name:
                
                label_col_name = CSV_LABEL_COLS[i] if i < len(CSV_LABEL_COLS) else f"Label Set {i+1}"
                no_label_text = f"No label specified in CSV column '{label_col_name}'."
                msg_label = ttk.Label(target_scrollable_frame, text=no_label_text, wraplength=GUI_IMAGE_DISPLAY_SIZE * (EXAMPLE_COLS -1))
                msg_label.grid(row=0, column=0, pady=5, padx=5, columnspan=EXAMPLE_COLS)
                continue 

            
            
            
            
            
            
            
            
            
            
            
            example_paths = self.example_label_paths_map.get(current_label_name, [])

            if not example_paths:
                
                if current_label_name in self.example_label_paths_map:
                    logging.warning(f"Label Set {i+1}: Label '{current_label_name}' (from CSV, padded) exists in example map but has no image paths (scanned from {EXAMPLE_DATA_ROOT}).")
                else:
                    logging.warning(f"Label Set {i+1}: Cannot find example images for label '{current_label_name}' (from CSV, padded). No subdirectory named '{current_label_name}' found or scanned in '{EXAMPLE_DATA_ROOT}'. Check if folder names match the padded format.")

            random.shuffle(example_paths)
            num_to_show = min(len(example_paths), NUM_EXAMPLE_IMAGES)

            if num_to_show == 0:
                no_examples_text = f"No example images found for label '{current_label_name}' in {EXAMPLE_DATA_ROOT}."
                
                msg_label = ttk.Label(target_scrollable_frame, text=no_examples_text, wraplength=GUI_IMAGE_DISPLAY_SIZE * (EXAMPLE_COLS -1))
                msg_label.grid(row=0, column=0, pady=5, padx=5, columnspan=EXAMPLE_COLS)
            else:
                
                for img_idx in range(num_to_show):
                    row, col = divmod(img_idx, EXAMPLE_COLS)
                    img_frame = ttk.Frame(target_scrollable_frame, padding=2)
                    img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                    img_label = ttk.Label(img_frame)
                    img_label.grid(row=0, column=0)
                    self._load_and_display_image(example_paths[img_idx], img_label, display_size=GUI_IMAGE_DISPLAY_SIZE)
                
                for c in range(EXAMPLE_COLS):
                    target_scrollable_frame.grid_columnconfigure(c, weight=1)


        
        self.pass_button.config(state=tk.NORMAL)
        self.fail_button.config(state=tk.NORMAL)

    def record_pass(self):
        
        self.pass_count += 1
        current_data_idx = self.selected_data_indices[self.current_test_index]
        current_img_path = self.all_test_image_paths[current_data_idx]
        
        log_msg = (f"Test {self.current_test_index + 1}/{self.num_tests}: PASS "
                   f"(User judged '{os.path.basename(current_img_path)}' visually correct "
                   f"for labels {self.current_gt_labels})")
        logging.info(log_msg)
        self._prepare_next_test()

    def record_fail(self):
        
        self.fail_count += 1
        current_data_idx = self.selected_data_indices[self.current_test_index]
        current_img_path = self.all_test_image_paths[current_data_idx]
        
        log_msg = (f"Test {self.current_test_index + 1}/{self.num_tests}: FAIL "
                   f"(User judged '{os.path.basename(current_img_path)}' visually incorrect "
                   f"for labels {self.current_gt_labels})")
        logging.info(log_msg)
        self._prepare_next_test()

    def _prepare_next_test(self):
        
        self.current_test_index += 1
        self.pass_button.config(state=tk.DISABLED)
        self.fail_button.config(state=tk.DISABLED)
        
        self.root.after(50, self.load_next_test)

    def show_final_results(self):
        
        total_judged = self.pass_count + self.fail_count
        
        accuracy = (self.pass_count / total_judged) * 100 if total_judged > 0 else 0

        print("\n--- Simple Visual Verification Results ---")
        print(f"Test Image Source: {CSV_FILE_PATH} (using column '{CSV_FILENAME_COL}')")
        print(f"Label Source: {CSV_FILE_PATH} (using columns {CSV_LABEL_COLS}, padded to {LABEL_PADDING_LENGTH} digits)") 
        print(f"Example Image Source: {EXAMPLE_DATA_ROOT}") 
        print(f"Total Images Tested (Judged by User): {total_judged}")
        print(f"Correct (Pass - User judged visually OK): {self.pass_count}")
        print(f"Incorrect (Fail - User judged visually wrong): {self.fail_count}")
        print(f"User-Judged Accuracy (Visual): {accuracy:.2f}%")
        print("------------------------------------------")

        result_message = (
            f"Visual Verification Complete!\n\n"
            f"Test Source: {CSV_FILE_PATH}\n"
            f"Label Source: {CSV_FILE_PATH} ({', '.join(CSV_LABEL_COLS)}, padded to {LABEL_PADDING_LENGTH} digits)\n" 
            f"Example Source: {EXAMPLE_DATA_ROOT}\n\n" 
            f"Total Images Judged: {total_judged}\n"
            f"Correct (Pass): {self.pass_count}\n"
            f"Incorrect (Fail): {self.fail_count}\n\n"
            f"User-Judged Accuracy: {accuracy:.2f}%"
        )
        messagebox.showinfo("Verification Results", result_message)
        self.root.quit()
        self.root.destroy()





if __name__ == "__main__":

    
    
    if not (len(CSV_LABEL_COLS) == NUM_LABEL_SETS):
        logging.error(f"Configuration error: The number of labels in CSV_LABEL_COLS ({len(CSV_LABEL_COLS)}) must equal NUM_LABEL_SETS ({NUM_LABEL_SETS}).")
        exit()

    
    test_paths, ground_truth_labels = load_data_from_csv(
        CSV_FILE_PATH,
        NUM_LABEL_SETS,
        CSV_FILENAME_COL,
        CSV_LABEL_COLS,
        TEST_IMAGE_BASE_DIR,
        pad_length=LABEL_PADDING_LENGTH 
    )
    if test_paths is None or ground_truth_labels is None:
        logging.error(f"Failed to load test data from CSV: {CSV_FILE_PATH}. Exiting.")
        exit()

    
    
    example_label_paths_map = get_all_example_image_paths(EXAMPLE_DATA_ROOT)
    if not example_label_paths_map and os.path.isdir(EXAMPLE_DATA_ROOT):
         logging.warning(f"No example label subdirectories or images found in the base example directory: {EXAMPLE_DATA_ROOT}. Examples section will be empty.")
         
    elif not os.path.isdir(EXAMPLE_DATA_ROOT):
         logging.error(f"The specified EXAMPLE_DATA_ROOT directory does not exist: {EXAMPLE_DATA_ROOT}. Cannot load examples.")
         
         


    
    if not test_paths: 
        logging.error(f"No test images loaded from {CSV_FILE_PATH}. Cannot start GUI.")
        exit()

    
    root = tk.Tk()
    app = SimpleImageVerifierGUI(root,
                                 example_label_paths_map, 
                                 test_paths, 
                                 ground_truth_labels, 
                                 NUM_LABEL_SETS,
                                 NUM_TESTS) 
    root.mainloop()

    logging.info("GUI closed. Visual verification finished.")
