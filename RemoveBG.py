import os
from PIL import Image, UnidentifiedImageError
from rembg import remove  # For background removal
from transformers import BlipProcessor, BlipForConditionalGeneration  # For AI captioning
import torch
import re  # For sanitizing filenames

# --- Configuration (Global Constants) ---
# Background Removal Parameters (from user's first script)
BG_USE_ALPHA_MATTING = True
BG_USE_POST_PROCESS_MASK = True
BG_ALPHA_MATTING_FG_THRESHOLD = 240
BG_ALPHA_MATTING_BG_THRESHOLD = 10
BG_ALPHA_MATTING_ERODE_SIZE = 15  # User's script had 15, rembg default is 10

# General Configuration
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
OUTPUT_SUBFOLDER_NAME = "processed_images_output"  # Name of the subfolder for processed images


# --- Helper Function: Rename Single Image with AI ---
def rename_single_image_with_ai(image_path_to_rename, processor, model, device, original_filename_for_log):
    """
    Renames a single image file based on AI-generated captions.
    The file is renamed in place.

    Args:
        image_path_to_rename (str): Full path to the image file to be renamed.
        processor: BlipProcessor instance.
        model: BlipForConditionalGeneration instance.
        device (str): "cuda" or "cpu".
        original_filename_for_log (str): Original filename for logging purposes.

    Returns:
        str: The new path of the renamed file if successful, None otherwise.
    """
    input_folder_path, filename_to_rename = os.path.split(image_path_to_rename)
    # rembg always outputs PNG, so the extension of the file to rename will be .png
    file_extension = ".png"

    try:
        # Open with 'with' to ensure the image is closed after use
        with Image.open(image_path_to_rename) as raw_image_pil:
            # BLIP model expects RGB images
            raw_image_rgb = raw_image_pil.convert("RGB")
            inputs = processor(raw_image_rgb, return_tensors="pt").to(device)

        # Generate caption using the AI model
        # Parameters like max_length, num_beams, early_stopping can be tuned for caption quality/length
        out = model.generate(**inputs, max_length=64, num_beams=5, early_stopping=True)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        print(f"    AI identified for '{original_filename_for_log}': '{caption}'")

        # Sanitize the caption to create a valid filename
        new_name_base = caption.lower().replace(' ', '_')
        new_name_base = re.sub(r'[^\w_.-]', '', new_name_base)  # Allow alphanumeric, underscore, dot, hyphen
        new_name_base = re.sub(r'_+', '_', new_name_base)  # Consolidate multiple underscores
        new_name_base = new_name_base.strip('_.- ')  # Remove leading/trailing junk characters

        if not new_name_base:  # If caption results in an empty string after sanitization
            new_name_base = "unidentifiable_image"

        # Limit the base name length to avoid issues with OS path length limits
        max_base_len = 100
        new_name_base = new_name_base[:max_base_len]

        counter = 0
        new_filename_attempt = f"{new_name_base}{file_extension}"
        new_image_path_attempt = os.path.join(input_folder_path, new_filename_attempt)

        # Handle potential duplicate filenames by adding a counter
        while os.path.exists(new_image_path_attempt):
            if new_image_path_attempt == image_path_to_rename:
                # This means the generated name is identical to the current name (e.g., "foo_no_bg.png" -> "foo_no_bg.png")
                # No actual rename is needed, consider it successfully "named".
                print(f"    -> Image '{filename_to_rename}' is already named appropriately or no change needed.")
                return image_path_to_rename

            counter += 1
            new_filename_attempt = f"{new_name_base}_{counter}{file_extension}"
            new_image_path_attempt = os.path.join(input_folder_path, new_filename_attempt)

        os.rename(image_path_to_rename, new_image_path_attempt)
        print(f"    -> Renamed '{filename_to_rename}' to '{new_filename_attempt}' in output folder.")
        return new_image_path_attempt

    except UnidentifiedImageError:
        print(
            f"    Error: Cannot identify image file '{filename_to_rename}'. It might be corrupted or not a valid image.")
        return None
    except Exception as e:
        print(f"    Error during AI renaming of '{filename_to_rename}': {e}")
        return None


# --- Main Processing Function ---
def process_images_in_folder(input_folder_path):
    """
    Processes images in a folder:
    1. Removes background and saves to an output subfolder.
    2. Renames the background-removed image using AI.
    3. Deletes the original image if steps 1 & 2 are successful.
    """
    output_folder_path = os.path.join(input_folder_path, OUTPUT_SUBFOLDER_NAME)

    # --- Setup AI Model (once) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for AI model: {device}")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print("AI model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load AI model: {e}")
        print("Please ensure 'transformers', 'torch', and 'Pillow' are installed.")
        print("An internet connection may be required for the first run to download the model.")
        return

    # --- Folder Setup ---
    if not os.path.isdir(input_folder_path):
        print(f"Error: Input folder not found at '{input_folder_path}'")
        return

    os.makedirs(output_folder_path, exist_ok=True)  # Create output subfolder if it doesn't exist
    print(f"\nStarting combined image processing in: '{input_folder_path}'")
    print(f"Processed images will be saved to: '{output_folder_path}'")
    print(f"Originals will be DELETED from '{input_folder_path}' upon successful processing of each image.")
    print("-" * 50)

    # Initialize counters for summary
    processed_count = 0
    bg_removal_error_count = 0
    rename_error_count = 0
    delete_error_count = 0
    skipped_files_count = 0
    skipped_dirs_count = 0

    items_in_input_folder = os.listdir(input_folder_path)

    for filename in items_in_input_folder:
        original_image_path = os.path.join(input_folder_path, filename)

        # Skip the output directory itself if it's listed
        if original_image_path == output_folder_path:
            continue

        if os.path.isdir(original_image_path):
            skipped_dirs_count += 1
            continue  # Skip other directories

        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            skipped_files_count += 1
            continue  # Skip non-image files

        print(f"\nProcessing '{filename}':")
        base_name, _ = os.path.splitext(filename)
        # Temporary filename for the background-removed image (always .png)
        no_bg_image_filename_temp = f"{base_name}_no_bg.png"
        path_to_no_bg_image_temp = os.path.join(output_folder_path, no_bg_image_filename_temp)

        # --- 1. Remove Background ---
        try:
            print(f"  1. Removing background...")
            with Image.open(original_image_path) as img_pil:
                # Convert to RGBA to handle various modes and ensure alpha channel for rembg
                img_for_rembg = img_pil.convert("RGBA")

                output_image_no_bg = remove(
                    img_for_rembg,
                    alpha_matting=BG_USE_ALPHA_MATTING,
                    post_process_mask=BG_USE_POST_PROCESS_MASK,
                    alpha_matting_foreground_threshold=BG_ALPHA_MATTING_FG_THRESHOLD,
                    alpha_matting_background_threshold=BG_ALPHA_MATTING_BG_THRESHOLD,
                    alpha_matting_erode_size=BG_ALPHA_MATTING_ERODE_SIZE
                )
            # output_image_no_bg is a PIL Image object, save it
            output_image_no_bg.save(path_to_no_bg_image_temp)
            print(f"     -> Background removed, saved temporarily as '{no_bg_image_filename_temp}' in output folder.")

        except UnidentifiedImageError:
            print(
                f"     Error: Cannot identify image file '{filename}'. It might be corrupted or not a valid image format for Pillow.")
            bg_removal_error_count += 1
            continue  # Skip to next file
        except Exception as e_bg:
            print(f"     Error removing background for '{filename}': {e_bg}")
            bg_removal_error_count += 1
            continue  # Skip to next file

        # --- 2. Rename with AI ---
        # The file to rename is the temporary background-removed image
        print(f"  2. Renaming '{no_bg_image_filename_temp}' with AI...")
        new_renamed_path = rename_single_image_with_ai(
            path_to_no_bg_image_temp,  # Full path to the temp _no_bg.png file
            processor,
            model,
            device,
            original_filename_for_log=filename  # Pass original filename for clearer logs
        )

        if new_renamed_path:
            # new_renamed_path is the full path to the successfully AI-renamed file in the output_folder_path
            print(
                f"     -> AI renaming successful. Final file in output folder: '{os.path.basename(new_renamed_path)}'")

            # --- 3. Delete Original File ---
            try:
                print(f"  3. Deleting original file '{original_image_path}'...")
                os.remove(original_image_path)
                print(f"     -> Original file deleted.")
                processed_count += 1
            except Exception as e_del:
                print(f"     Error deleting original file '{original_image_path}': {e_del}")
                print(f"     PLEASE MANUALLY DELETE '{original_image_path}' IF DESIRED.")
                delete_error_count += 1
        else:
            # AI renaming failed, the _no_bg.png file remains in the output folder with its temporary name.
            print(f"     AI renaming failed for '{no_bg_image_filename_temp}'. Original file '{filename}' NOT deleted.")
            rename_error_count += 1

    # --- Summary ---
    print("-" * 50)
    print("Image processing complete!")
    print(f"Successfully processed (BG removed, AI renamed, original deleted): {processed_count}")
    print(f"Errors during background removal: {bg_removal_error_count}")
    print(f"Errors during AI renaming (original not deleted for these): {rename_error_count}")
    print(f"Errors during original file deletion (after successful BG removal & rename): {delete_error_count}")
    print(f"Non-image files skipped: {skipped_files_count}")
    print(f"Directories skipped (excluding the output subfolder): {skipped_dirs_count}")
    print(f"Check the '{output_folder_path}' folder for the final results.")
    print("-" * 50)


# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT: Set your target folder path here ---
    # This is the folder containing the ORIGINAL images.
    # Processed images will go into a subfolder, and originals will be deleted from here.

    # Get target folder from user input
    default_target_folder = r"C:\Path\To\Your\Image\Folder"  # Example placeholder
    print("This script processes images by removing backgrounds, renaming them with AI,")
    print("and then deleting the original files if all steps are successful.")
    print("\nPlease specify the full path to the folder containing your original images.")
    print(f"Example: {default_target_folder}")

    target_input_folder_str = input("Enter target folder path: ").strip()

    if not target_input_folder_str:
        print("No folder path entered. Exiting.")
    elif not os.path.isdir(target_input_folder_str):
        print(f"Error: The path '{target_input_folder_str}' is not a valid directory. Exiting.")
    else:
        # Confirmation step before destructive operation
        confirmation_message = (
            f"\nWARNING: This script will process images in:\n  '{target_input_folder_str}'\n\n"
            f"1. Backgrounds will be removed.\n"
            f"2. Images will be saved to a subfolder named '{OUTPUT_SUBFOLDER_NAME}' within it.\n"
            f"3. These saved images will be renamed based on AI-generated captions.\n"
            f"4. If steps 1-3 are successful for an image, THE ORIGINAL FILE in '{target_input_folder_str}' WILL BE DELETED.\n\n"
            "It is STRONGLY recommended to BACK UP your original images before proceeding.\n\n"
            "Are you sure you want to continue? (yes/no): "
        )
        confirm = input(confirmation_message).strip().lower()

        if confirm == 'yes':
            process_images_in_folder(target_input_folder_str)
        else:
            print("Operation cancelled by user.")