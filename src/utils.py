import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pillow_heif
from PIL.ExifTags import TAGS
import piexif
from pillow_heif import register_heif_opener

def decode_exif(exif_bytes):

    exif_data = {}
    if not exif_bytes:
        return exif_data  # No EXIF data found, return empty dictionary

    try:
        exif_dict = piexif.load(exif_bytes)

        # Flatten the EXIF data from all IFDs
        for ifd in exif_dict:
            for tag, value in exif_dict[ifd].items():
                decoded_tag = TAGS.get(tag, tag)
                exif_data[decoded_tag] = value

    except Exception:
        pass  # Suppress errors silently
    return exif_data

def convert_exif_value(tag, value):

    if value is None:
        return None

    try:
        if tag in ["ExposureTime", "FNumber", "BrightnessValue", "ExposureBiasValue"]:
            if isinstance(value, tuple) and len(value) == 2:
                numerator, denominator = value
                if denominator != 0:
                    return numerator / denominator
                else:
                    return None
            elif isinstance(value, int):
                return float(value)
            elif isinstance(value, float):
                return value
            else:
                return None
        elif tag == "ISOSpeedRatings":
            if isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                return int(value[0])
            else:
                return None
        else:
            return None
    except Exception:
        return None

def calculate_calculated_exposure(exposure_time, f_number, iso):

    if exposure_time is None or f_number is None or iso is None:
        return None
    try:
        if iso == 0:
            return None  # Prevent division by zero
        return exposure_time * iso / (f_number**2)
    except Exception:
        return None
    
def estimate_lux(aperture, shutter_speed, iso, image):

    # 1) Validate inputs
    if not aperture or not shutter_speed or not iso:
        return None, None  # Cannot compute lux if any parameter is missing/invalid
    if iso == 0:
        return None, None  # Prevent divide by zero

    # 2) Calculate unadjusted Lux
    #    Lux_unadjusted ~= 2.5 * (aperture^2 * 100) / (shutter_speed * iso)
    lux_unadjusted = 2.5 * (aperture ** 2) * 100 / (shutter_speed * iso)

    # 3) Compute mean pixel value
    #    - If your image is BGR, convert to grayscale or just average across all channels
    #    - range: 0-255
    #    - We'll just average all channels to get a single brightness value
    mean_pixel_val = float(np.mean(image))

    # 4) Calculate adjusted Lux
    #    Lux_adjusted ~= Lux_unadjusted * (mean_pixel_val / 128)
    
    gamma = 2.2
    lux_adjusted = lux_unadjusted * (mean_pixel_val ** gamma) / (128 ** gamma)

    return lux_unadjusted, lux_adjusted, mean_pixel_val

def extract_specific_metadata(image_paths):
    """
    Extracts metadata for each image path in 'image_paths'.
    Additionally, calculates two lux estimates (unadjusted and adjusted by mean pixel value)
    using the 'estimate_lux' function.
    """

    # Define the EXIF tags to extract
    desired_tags = [
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
        "BrightnessValue",
        "ExposureBiasValue"
    ]

    metadata_list = []

    for image_path in image_paths:
        metadata = {"filename": image_path}
        try:
            # Read HEIC file and metadata
            heif_file = pillow_heif.read_heif(image_path)
            exif_bytes = heif_file.info.get("exif", b"")

            # Decode EXIF bytes
            decoded_exif = decode_exif(exif_bytes)

            # Extract and convert only the desired tags
            extracted_values = {}
            for tag in desired_tags:
                raw_value = decoded_exif.get(tag, None)
                converted_value = convert_exif_value(tag, raw_value)
                extracted_values[tag] = converted_value
                metadata[tag] = converted_value

            # Calculate calculatedExposure (as before)
            exposure_time = extracted_values.get("ExposureTime")
            f_number = extracted_values.get("FNumber")
            iso = extracted_values.get("ISOSpeedRatings")
            calculated_exposure = calculate_calculated_exposure(exposure_time, f_number, iso)
            metadata["calculatedExposure"] = calculated_exposure

            # 1) Load the image as a NumPy array (BGR)
            image = load_image_as_opencv(image_path)

            if (image is not None 
                and exposure_time is not None 
                and f_number is not None 
                and iso is not None):

                # 2) Call estimate_lux to get unadjusted and adjusted lux values
                lux_unadjusted, lux_adjusted, mean_pixel_value = estimate_lux(
                    aperture=f_number,
                    shutter_speed=exposure_time,
                    iso=iso,
                    image=image
                )
                metadata["lux_unadjusted"] = lux_unadjusted
                metadata["lux_adjusted"] = lux_adjusted
                metadata["mean_pixel_value"] = mean_pixel_value
            else:
                # If something is missing or image loading failed
                metadata["lux_unadjusted"] = None
                metadata["lux_adjusted"] = None
                metadata["mean_pixel_value"] = None

        except Exception as e:
            print(f"Error processing file {image_path}: {e}")
            # If there's an error, set all desired tags and calculatedExposure to None
            for tag in desired_tags:
                metadata[tag] = None
            metadata["calculatedExposure"] = None
            metadata["lux_unadjusted"] = None
            metadata["lux_adjusted"] = None
            metadata["mean_pixel_value"] = None

        metadata_list.append(metadata)

    return metadata_list

register_heif_opener()

def load_image_as_opencv(filename):
    """
    Loads an image from disk and returns it as a BGR numpy array.
    This function handles HEIC (or HEIF) by using Pillow + pillow-heif under the hood.
    """
    try:
        # Open with Pillow
        pil_image = Image.open(filename).convert("RGB")  # ensure we have an RGB image
        # Convert to numpy array (RGB)
        rgb_array = np.array(pil_image, dtype=np.uint8)
        # Convert from RGB (Pillow) to BGR (OpenCV)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        return None

def load_image_as_opencv(filename):
    """
    Loads an image from disk and returns it as a BGR numpy array.
    This function handles HEIC (or HEIF) by using Pillow + pillow-heif under the hood.
    """
    try:
        # Open with Pillow
        pil_image = Image.open(filename).convert("RGB")  # Ensure we have an RGB image
        # Convert to numpy array (RGB)
        rgb_array = np.array(pil_image, dtype=np.uint8)
        # Convert from RGB (Pillow) to BGR (OpenCV)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        return None

def create_hdr_from_images(
    image_paths,
    output_debevec="hdr_debevec.jpg"
):

    
    # 1) Extract metadata to get the exposure times
    metadata_list = extract_specific_metadata(image_paths)

    # 2) Read images and collect exposures
    images = []
    exposure_times = []
    dimensions = []

    for meta in metadata_list:
        filename = meta["filename"]
        
        # Load the image using our universal loader
        img = load_image_as_opencv(filename)
        if img is None:
            raise IOError(f"Could not read image (HEIC or otherwise): {filename}")
        
        images.append(img)
        dimensions.append(img.shape[:2])  # (height, width)
        print(f"Loaded {filename} with dimensions: {img.shape}")
        
        # Use the 'calculatedExposure' as the exposure time
        # Make sure it's a float32 array for OpenCV
        exposure_times.append(float(meta["calculatedExposure"]))

    # 3) Ensure all images have the same dimensions by scaling them to the smallest size
    unique_dimensions = set(dimensions)
    if len(unique_dimensions) != 1:
        print("Images have varying dimensions. Resizing to a common size.")

        # Extract minimum height and width to scale down to
        min_height = min([dim[0] for dim in dimensions])
        min_width = min([dim[1] for dim in dimensions])
        print(f"Target dimensions for resizing: (Height: {min_height}, Width: {min_width})")

        resized_images = []
        for idx, img in enumerate(images):
            original_height, original_width = img.shape[:2]
            if (original_height, original_width) != (min_height, min_width):
                # Perform resizing using INTER_AREA for downscaling
                resized_img = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
                resized_images.append(resized_img)
                print(f"Resized {image_paths[idx]} from ({original_height}, {original_width}) to ({min_height}, {min_width})")
            else:
                # No resizing needed
                resized_images.append(img)
                print(f"No resizing needed for {image_paths[idx]}")
        
        images = resized_images
    else:
        print("All images have the same dimensions. No resizing needed.")

    # 4) Convert exposure_times to the shape and type OpenCV expects
    exposure_times = np.array(exposure_times, dtype=np.float32)

    # 5) Calibrate and merge with Debevec
    print("Calibrating and merging with Debevec...")
    calibrate_debevec = cv2.createCalibrateDebevec()
    response_debevec = calibrate_debevec.process(images, exposure_times)
    
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(images, exposure_times, response_debevec)

    # 6) Linearly scale the HDR image to 8-bit range (0-255)
    print("Scaling HDR image to 8-bit range...")
    # Find the minimum and maximum radiance values
    hdr_min = np.min(hdr_debevec)
    hdr_max = np.max(hdr_debevec)
    
    print(f"HDR Image - Min Radiance: {hdr_min}, Max Radiance: {hdr_max}")

    # Avoid division by zero
    if hdr_max - hdr_min == 0:
        print("Warning: HDR image has zero dynamic range. Creating a black image.")
        ldr_debevec_8bit = np.zeros_like(hdr_debevec, dtype=np.uint8)
    else:
        # Normalize HDR image to 0-255
        normalized_hdr = (hdr_debevec - hdr_min) / (hdr_max - hdr_min)  # Normalize to 0-1
        scaled_hdr = normalized_hdr * 255  # Scale to 0-255
        ldr_debevec_8bit = scaled_hdr.astype('uint8')  # Convert to 8-bit

    # 7) Save the resulting 8-bit HDR image
    cv2.imwrite(output_debevec, ldr_debevec_8bit)
    print(f"Saved Debevec HDR to: {output_debevec}")