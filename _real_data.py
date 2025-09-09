import os
import glob
import random
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
import numpy as np
import cv2


# step size in degrees (can set 0.5 if you want denser coverage)
ROT_STEP = 1.0
NUM_AUGS = 3
MIN_SAMPLES_PER_CLASS = 70
MAX_ATTEMPTS_PER_SYMBOL = 3000

# 1. SETTINGS

IMAGE_SIZE = (384, 384)
PADDING = int(IMAGE_SIZE[0] * 0.4)
IMAGE_SIZE = (IMAGE_SIZE[0] + PADDING * 2, IMAGE_SIZE[1] + PADDING * 2)

DATASET_DIR = "symbols_dataset"
FONTS_DIR = "fonts"
LABEL_FILE = os.path.join(DATASET_DIR, "labels.txt")
DESCRIPTION_FILE = os.path.join(DATASET_DIR, "symbol_descriptions.txt")

# 2. SYMBOL SETS

GEOMETRIC_SYMBOLS = {
    "↗": "Runout",
    "↧": "Depth",
    "∥": "Parallelism",
    "⊕": "Position (RFS)",
    "⊥": "Perpendicularity",
    "⌀": "Diameter",
    "⌒": "Profile of a Line",
    "⌓": "Profile of a Surface",
    "⌖": "True Position",
    "⌭": "Cylindricity",
    "⌯": "Symmetry",
    "⌰": "Total Runout",
    "⌴": "Counterbore",
    "⌵": "Countersink",
    "▱": "Flatness",
    "◎": "Concentricity",
    "◯": "Circularity",
    "⦹": "Circled Perpendicularity",
}

ALL_SYMBOLS = sorted(set(list(GEOMETRIC_SYMBOLS.keys())))
SYMBOL_DESCRIPTIONS = {**GEOMETRIC_SYMBOLS}


# 3. UTILITIES


def is_char_supported(font_path, char):
    try:
        font = ImageFont.truetype(font_path, size=120)
        mask = font.getmask(char)
        return mask.size != (0, 0)
    except Exception:
        return False


def is_box_shaped(img):
    img_cv = np.array(img)
    _, thresh = cv2.threshold(img_cv, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = h * w
    img2 = np.array(img.copy())
    cv2.drawContours(img2, [largest], 0, (0), -1)
    _, thresh2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY_INV)
    contours2, _ = cv2.findContours(
        thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        return True
    largest2 = max(contours2, key=cv2.contourArea)
    filled_area = cv2.contourArea(largest2)
    return (bbox_area - filled_area) < 0.10 * bbox_area


def generate_image(char, font_path, font_size=None, margin=50):
    try:
        if font_size is None:
            font_size = random.randint(150, 300)
        font = ImageFont.truetype(font_path, size=font_size)
        if not is_char_supported(font_path, char):
            return None
    except Exception:
        return None

    img = Image.new("L", IMAGE_SIZE, color=255)
    draw = ImageDraw.Draw(img)

    # Get text size
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Ensure safe placement with extra margin from edges
    safe_x_min = PADDING + margin
    safe_y_min = PADDING + margin
    safe_x_max = IMAGE_SIZE[0] - w - PADDING - margin
    safe_y_max = IMAGE_SIZE[1] - h - PADDING - margin

    # If the font is too big to fit with margin, reduce size dynamically
    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        return None

    pos_x = random.randint(safe_x_min, safe_x_max)
    pos_y = random.randint(safe_y_min, safe_y_max)
    pos = (pos_x, pos_y)

    # Render symbol
    draw.text(pos, char, font=font, fill=20)

    if is_box_shaped(img):
        return None
    return img


def add_realistic_noise(img):
    img_array = np.array(img)
    if random.random() < 0.3:
        noise_density = random.uniform(0.001, 0.01)
        num_salt = np.ceil(noise_density * img_array.size * 0.5)
        num_pepper = np.ceil(noise_density * img_array.size * 0.5)
        coords = [np.random.randint(0, i-1, int(num_salt))
                  for i in img_array.shape]
        img_array[tuple(coords)] = 255
        coords = [np.random.randint(0, i-1, int(num_pepper))
                  for i in img_array.shape]
        img_array[tuple(coords)] = 0
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(5, 20), img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def add_motion_blur(img):
    if random.random() < 0.25:
        img_array = np.array(img)
        size = random.randint(3, 8)
        kernel_motion_blur = np.zeros((size, size))
        if random.random() < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        img_array = cv2.filter2D(img_array, -1, kernel_motion_blur)
        return Image.fromarray(img_array)
    return img


def add_dilation(img):
    """Add morphological dilation to simulate bold/bloated symbols or ink bleed"""
    if random.random() < 0.3:  # 30% chance of applying dilation
        img_array = np.array(img)

        # Create structuring element (kernel) - elliptical shape works well for symbols
        # Small kernel to avoid over-dilation
        kernel_size = random.randint(2, 4)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Apply dilation - this makes the symbol strokes thicker
        dilated = cv2.dilate(img_array, kernel, iterations=1)

        return Image.fromarray(dilated)
    return img


def add_gaussian_blur(img):
    """Add Gaussian blur with dark edge enhancement for realistic halos"""
    img_array = np.array(img).astype(np.float32)
    img_inv = 255 - img_array  # Work in ink space
    img_inv = np.clip(img_inv * 1.2, 0, 255)  # Boost ink density
    radius = random.uniform(0.8, 3.0)  # Stronger blur for visible halos
    img_inv_pil = Image.fromarray(img_inv.astype(np.uint8))
    img_inv_blur = img_inv_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    img_blur = 255 - np.array(img_inv_blur)
    img_blur = np.clip(img_blur, 0, 255)
    pil_img = Image.fromarray(img_blur.astype(np.uint8))
    # Mild contrast boost to make symbol pop
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.1)
    return pil_img


def add_perspective_transform(img):
    if random.random() < 0.2:
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        shift = random.randint(5, 20)
        dst_points = np.float32([
            [random.randint(0, shift), random.randint(0, shift)],
            [w - random.randint(0, shift), random.randint(0, shift)],
            [random.randint(0, shift), h - random.randint(0, shift)],
            [w - random.randint(0, shift), h - random.randint(0, shift)]
        ])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img_array = cv2.warpPerspective(
            img_array, matrix, (w, h), borderValue=255)
        return Image.fromarray(img_array)
    return img


def add_scan_artifacts(img):
    if random.random() < 0.15:
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        if random.random() < 0.5:
            line_y = random.randint(0, h-1)
            intensity = random.randint(180, 220)
            img_array[line_y:line_y+1, :] = intensity
        else:
            line_x = random.randint(0, w-1)
            intensity = random.randint(180, 220)
            img_array[:, line_x:line_x+1] = intensity
        return Image.fromarray(img_array)
    return img


def add_subtle_vignette(img):
    """Add very subtle darkening at corners to simulate lens/scan falloff"""
    if random.random() < 0.3:
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w // 2, h // 2
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        # 30% max darkening at corners
        vignette = 1 - (distance / max_dist * 0.3)
        img_array = img_array * vignette
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    return img


def advanced_augment_image(img):
    # ROTATION: ONLY ±70 DEGREES
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)  # any float between –10° and +10°
        img = img.rotate(angle, fillcolor=255)

    if random.random() < 0.4:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.4))

    # GAUSSIAN BLUR + OTHER EFFECTS
    img = add_gaussian_blur(img)      # Always applied, with dark halos
    img = add_motion_blur(img)
    img = add_realistic_noise(img)
    img = add_perspective_transform(img)
    img = add_scan_artifacts(img)
    # NEW: Apply dilation to simulate bold symbols
    img = add_dilation(img)

    # Random scaling
    if random.random() < 0.3:
        scale_factor = random.uniform(0.5, 1.0)
        new_size = (int(img.width * scale_factor),
                    int(img.height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)

    # ADD VIGNETTE (NEW)
    img = add_subtle_vignette(img)

    return img

# 4. CLEAN & SETUP


if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)

# 5. LOAD FONTS

font_paths = glob.glob(os.path.join(FONTS_DIR, "*.ttf"))
if not font_paths:
    print(f"No .ttf fonts found in '{FONTS_DIR}/'. Please add some.")
    print("Tip: Download 'Symbola.ttf' for full GD&T and symbol support.")
    exit(1)


# 6. GENERATE DATA

label_entries = set()
print(
    f"Generating ≥{MIN_SAMPLES_PER_CLASS} per class ({len(ALL_SYMBOLS)} symbols)"
)

num_fonts = len(font_paths)
samples_per_font = MIN_SAMPLES_PER_CLASS // num_fonts

for char in tqdm(ALL_SYMBOLS, desc="Symbols"):
    safe_char = f"U{ord(char):04X}"
    char_dir = os.path.join(DATASET_DIR, safe_char)
    os.makedirs(char_dir, exist_ok=True)

    sample_id = 0

# helper: apply small continuous random rotation

    def apply_small_rotation(img):
        angle = random.uniform(-10, 10)  # random float tilt
        return img.rotate(angle, fillcolor=255)

    # cycle through each font equally
    for font_path in font_paths:
        for i in range(samples_per_font):
            # generate original
            img_original = generate_image(char, font_path)
            if img_original is None:
                continue
            img_original = apply_small_rotation(img_original)  # random tilt

            # 1. SAVE ORIGINAL
            filename = f"{safe_char}_{sample_id:05}_orig.png"
            img_path = os.path.join(char_dir, filename)
            img_original.save(img_path)
            label_entries.add(f"{safe_char}/{filename}\t{char}")
            sample_id += 1

            # 2. SAVE MULTIPLE RANDOM POSITIONS
            for pos_variant in range(2):
                img_variant = generate_image(char, font_path)
                if img_variant is None:
                    continue
                img_variant = apply_small_rotation(img_variant)  # random tilt
                filename = f"{safe_char}_{sample_id:05}_pos{pos_variant}.png"
                img_path = os.path.join(char_dir, filename)
                img_variant.save(img_path)
                label_entries.add(f"{safe_char}/{filename}\t{char}")
                sample_id += 1

            # 3. SAVE AUGMENTED VERSIONS
            for aug_id in range(NUM_AUGS):
                img_aug = advanced_augment_image(img_original)
                img_aug = apply_small_rotation(img_aug)  # random tilt
                filename = f"{safe_char}_{sample_id:05}_aug{aug_id}.png"
                img_path = os.path.join(char_dir, filename)
                img_aug.save(img_path)
                label_entries.add(f"{safe_char}/{filename}\t{char}")
                sample_id += 1

            # 4. SYSTEMATIC ROTATION SWEEP
            # e.g., -10, -9, ... 0, +9, +10
            for angle in np.arange(-10, 10.1, ROT_STEP):
                img_rot = img_original.rotate(angle, fillcolor=255)
                filename = f"{safe_char}_{sample_id:05}_rot{angle:+.1f}.png"
                img_path = os.path.join(char_dir, filename)
                img_rot.save(img_path)
                label_entries.add(f"{safe_char}/{filename}\t{char}")
                sample_id += 1

# 7. BALANCE DATASET


def count_samples_per_class():
    """Count current samples per class"""
    counts = {}
    for char in ALL_SYMBOLS:
        safe_char = f"U{ord(char):04X}"
        char_dir = os.path.join(DATASET_DIR, safe_char)
        if os.path.exists(char_dir):
            counts[char] = len(
                [f for f in os.listdir(char_dir) if f.endswith('.png')])
        else:
            counts[char] = 0
    return counts


def get_next_sample_id(char_dir):
    """Get the next available sample ID for a class directory"""
    existing_files = [f for f in os.listdir(char_dir) if f.endswith('.png')]
    if not existing_files:
        return 0

    max_id = 0
    for filename in existing_files:
        # Extract numeric part from filename like U2299_00123_orig.png
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                id_part = parts[1]
                # Handle cases like 00123_orig
                id_num = int(id_part.split('_')[0])
                max_id = max(max_id, id_num)
            except (ValueError, IndexError):
                continue
    return max_id + 1


def generate_additional_samples(char, target_count, current_count):
    """Generate additional samples to reach target count"""
    safe_char = f"U{ord(char):04X}"
    char_dir = os.path.join(DATASET_DIR, safe_char)

    needed_samples = target_count - current_count
    if needed_samples <= 0:
        return 0

    generated_count = 0
    sample_id = get_next_sample_id(char_dir)

    # Keep generating until we have enough or hit max attempts
    attempts = 0
    max_attempts = needed_samples * 10  # Allow some failures

    with tqdm(total=needed_samples, desc=f"Balancing {char}", leave=False) as pbar:
        while generated_count < needed_samples and attempts < max_attempts:
            # Pick a random font
            font_path = random.choice(font_paths)

            # Generate base image
            img_original = generate_image(char, font_path)
            if img_original is None:
                attempts += 1
                continue

            # Apply random rotation
            angle = random.uniform(-10, 10)
            img_original = img_original.rotate(angle, fillcolor=255)

            # Generate different types of variations
            variation_types = ['pos', 'aug', 'rot']
            variation_type = random.choice(variation_types)

            if variation_type == 'pos':
                # Random position variant
                img_variant = generate_image(char, font_path)
                if img_variant is not None:
                    angle = random.uniform(-10, 10)
                    img_variant = img_variant.rotate(angle, fillcolor=255)
                    filename = f"{safe_char}_{sample_id:05}_bal_pos.png"
                    img_path = os.path.join(char_dir, filename)
                    img_variant.save(img_path)
                    label_entries.add(f"{safe_char}/{filename}\t{char}")
                    generated_count += 1
                    sample_id += 1
                    pbar.update(1)

            elif variation_type == 'aug':
                # Augmented variant
                img_aug = advanced_augment_image(img_original)
                angle = random.uniform(-10, 10)
                img_aug = img_aug.rotate(angle, fillcolor=255)
                filename = f"{safe_char}_{sample_id:05}_bal_aug.png"
                img_path = os.path.join(char_dir, filename)
                img_aug.save(img_path)
                label_entries.add(f"{safe_char}/{filename}\t{char}")
                generated_count += 1
                sample_id += 1
                pbar.update(1)

            elif variation_type == 'rot':
                # Rotated variant
                rot_angle = random.uniform(-10, 10)
                img_rot = img_original.rotate(rot_angle, fillcolor=255)
                filename = f"{safe_char}_{sample_id:05}_bal_rot{rot_angle:+.1f}.png"
                img_path = os.path.join(char_dir, filename)
                img_rot.save(img_path)
                label_entries.add(f"{safe_char}/{filename}\t{char}")
                generated_count += 1
                sample_id += 1
                pbar.update(1)

            attempts += 1

    return generated_count


print("\n Starting dataset balancing...")

# Progressive balancing loop
balance_round = 1
while True:
    print(f"\n Balance Round {balance_round}")

    # Count current samples
    counts = count_samples_per_class()
    max_count = max(counts.values())
    min_count = min(counts.values())

    print(f"Current range: {min_count} - {max_count} samples")

    # Check if already balanced
    if min_count == max_count:
        print("Dataset is perfectly balanced!")
        break

    print(f"Balancing to {max_count} samples per class...")

    # Generate additional samples for under-represented classes
    for char in tqdm(ALL_SYMBOLS, desc="Classes"):
        current_count = counts[char]
        if current_count < max_count:
            added = generate_additional_samples(char, max_count, current_count)
            if added > 0:
                print(f"  Added {added} samples for '{char}'")

    balance_round += 1

    # Safety check to prevent infinite loops
    if balance_round > 10:
        print("Maximum balance rounds reached. Stopping.")
        break

# 8. SAVE METADATA
with open(LABEL_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(label_entries)))

with open(DESCRIPTION_FILE, "w", encoding="utf-8") as f:
    for char in ALL_SYMBOLS:
        desc = SYMBOL_DESCRIPTIONS.get(char, "Unknown")
        f.write(f"U{ord(char):04X}\t{char}\t{desc}\n")

# Final verification
final_counts = count_samples_per_class()
final_max = max(final_counts.values())
final_min = min(final_counts.values())

print("\n" + "="*50)
print("FINAL DATASET STATISTICS")
print("="*50)
print(f"Total symbols: {len(ALL_SYMBOLS)}")
print(f"Samples per class: {final_max}")
print(f"Range: {final_min} - {final_max}")
print(f"Labels saved to: {LABEL_FILE}")
print(f"Dataset saved to: {DATASET_DIR}")
print("="*50)
