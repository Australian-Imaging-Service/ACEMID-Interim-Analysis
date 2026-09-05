from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import string
import subprocess
from PIL import Image

CLEANSED_DATA = Path("/Users/user/script_2/cleansed_data")
output_folder = Path("/Users/user/script_2/output")
MAX_THREADS = 8


def find_baseline_folder(patient_folder):
    """
    Find the subfolder associated with the baseline capture
    We will assume that the baseline subfolder is the earliest subfolder
    (alphabetically) with at least 44 Canon raw files. In theory, there should be
    either 46 or 92 Canon raw files in the baseline folder (depending on version of
    the WB360) - but we'll use 44 as a lower bound to account for broken cameras,
    missing files, etc.
    """
    print(f"Searching for baseline folder in {patient_folder}")

    canon_exts = (".crw", ".cr2", ".cr3")
    baseline_folder = None
    for subfolder in sorted(
        [item for item in patient_folder.iterdir() if item.is_dir()]
    ):
        canon_raw_files = [
            x for x in subfolder.glob("*") if x.suffix.lower() in canon_exts
        ]
        if len(canon_raw_files) >= 44:
            baseline_folder = subfolder
            print(f"\tFound baseline folder: {baseline_folder}")
            break
    return baseline_folder


def get_camera_model(baseline_folder):
    """
    Open the first "B" CR2 or CR3 file in the baseline folder and read the camera model
    from the EXIF data.
    """
    raw_images = list(baseline_folder.glob("*B.CR2")) + list(
        baseline_folder.glob("*B.CR3")
    )
    if not raw_images:
        return None

    with Image.open(raw_images[0]) as im:
        exif = im.getexif()
        camera_model = exif.get(0x0110)  # EXIF model tag
        return camera_model


def convert_raw_images(baseline_folder, output_folder, use_wine=False):
    """
    Convert each CR2 or CR3 file in the baseline capture folder to PNG or JPG.
    Each image conversion runs in a separate worker thread because the underlying
    subprocess calls are independent and can be done in parallel.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # First, check whether these are CR2 or CR3 files
    cr2s = list(baseline_folder.glob("*B.CR2"))
    cr3s = list(baseline_folder.glob("*B.CR3"))
    raw_files = cr2s + cr3s

    if not raw_files:
        return []

    raw_type = "CR3" if len(cr3s) > len(cr2s) else "CR2"
    print("Detected raw type:", raw_type)

    # Get the camera model from the first raw file
    camera_model = get_camera_model(baseline_folder)
    print("Detected camera model:", camera_model)

    converters = {
        "v1": Path(__file__).parent / "raw_converter_v1" / "convertCR2.exe",
        "v2": Path(__file__).parent / "raw_converter_v2" / "WbRawToPng.exe",
    }

    def check_fix_orientation_inplace(image_path):
        image_path = Path(image_path)

        rot90ccw = [
            "a1B",
            "a2B",
            "a5B",
            "a10B",
            "a13B",
            "a14B",
            "f1B",
            "f2B",
            "f5B",
            "f10B",
            "f13B",
            "f14B",
        ]
        print("Checking orientation for: ", image_path)
        image_id = image_path.stem.split("_")[-1]
        with Image.open(image_path) as im:
            width, height = im.size
            # check if image is already in portrait orientation (height > width)
            if height > width:
                print(f"\tAlready in portrait, skipping")
            else:
                rotation = (
                    Image.Transpose.ROTATE_90
                    if image_id in rot90ccw
                    else Image.Transpose.ROTATE_270
                )
                print(
                    f"\tRotating {image_path.name} CCW by {Image.Transpose(rotation).name} deg"
                )
                im = im.transpose(rotation)
                im.save(image_path, format="PNG")

    def convert_one(raw_file):
        print(f"Converting to PNG: {raw_file.stem}")
        infile = str(raw_file)
        timestamp = Path(infile).parent.name
        guid = Path(infile).parent.parent.name
        outfile = str(output_folder / f"{guid}_{timestamp}_{raw_file.stem}.png")
        if Path(outfile).exists():
            print(f"Output file already exists, skipping: {outfile}")
            check_fix_orientation_inplace(outfile)
            return outfile

        converter = (
            converters["v1"]
            if camera_model in ("Canon EOS Rebel T6", "Canon EOS 6D")
            else converters["v2"]
        )
        prefix = ["wine"] if use_wine else []
        suffix = ["-m", "xp"] if raw_type == "CR3" else []
        args = prefix + [str(converter), "-i", infile, "-o", outfile] + suffix

        subprocess.run(args, check=True)

        check_fix_orientation_inplace(outfile)

        return outfile

    converted_images = []
    max_workers = min(MAX_THREADS, len(raw_files))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_one, raw_file) for raw_file in raw_files]
        for future in futures:
            converted_images.append(future.result())

    return converted_images


def create_tiles(converted_image_list, output_folder):

    alphabet = list(string.ascii_lowercase)
    tile_count_x = 5
    tile_count_y = 9
    tiles = []

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    def tile_one_image(converted_image):
        with Image.open(converted_image) as im:
            print(f"Creating {tile_count_x}x{tile_count_y} tiles for {converted_image}")
            image_width, image_height = im.size
            source_image = Path(converted_image).stem

            tile_width = image_width / tile_count_x
            tile_height = image_height / tile_count_y
            generated_tiles = []

            for i in range(tile_count_x):
                for j in range(tile_count_y):
                    left = round(i * tile_width)
                    upper = round(j * tile_height)
                    right = round((i + 1) * tile_width)
                    lower = round((j + 1) * tile_height)

                    alpha = alphabet[i]
                    output_filename = f"{source_image}_{alpha}{j+1}.jpg"
                    output_path = output_folder / output_filename

                    tile = im.crop((left, upper, right, lower))
                    tile.save(output_path, format="JPEG", quality=95, subsampling=0)
                    generated_tiles.append(output_path)

        return generated_tiles

    max_workers = (
        min(MAX_THREADS, len(converted_image_list)) if converted_image_list else 1
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_tiles in executor.map(tile_one_image, converted_image_list):
            tiles.extend(image_tiles)

    return tiles


def perform_tile_qc(tile_list):

    MIN_SKIN_PROPORTION = 0.33

    rejects = (
        "a1B_b9",
        "a1B_d1",
        "a1B_d2",
        "a1B_d3",
        "a1B_d4",
        "a1B_d5",
        "a1B_e1",
        "a1B_e2",
        "a1B_e3",
        "a1B_e4",
        "a1B_e5",
        "a1B_e6",
        "a1B_e7",
        "a1B_e8",
        "a1B_e9",
        "a14B_1a",
        "a14B_a2",
        "a14B_a3",
        "a14B_a4",
        "a14B_a5",
        "a14B_a6",
        "a14B_a7",
        "a14B_b2",
        "a14B_d9",
    )

    qc = []
    print("Performing tile QC...")

    for tile_image in tile_list:
        # filename will be in form <guid>_<timestamp>_<imageid>_<tileid>.ext
        # just get the <imageid>_<tileid> part of the filename
        tile_identifier = "_".join(Path(tile_image).stem.split("_")[-2:])

        if tile_identifier in rejects:
            print(f"QC Fail (reason: in reject list): {tile_image.name}")
            qc.append(
                {
                    "tile_image": tile_image,
                    "result": "reject",
                    "reason": "tile is in reject list",
                }
            )
            continue

        im = Image.open(tile_image).convert("RGB")
        hsv_im = im.convert("HSV")
        ycbcr_im = im.convert("YCbCr")
        skin_pixels = 0
        total_pixels = im.width * im.height
        for hsv_px, ycbcr_px in zip(hsv_im.getdata(), ycbcr_im.getdata()):
            h, s, v = hsv_px
            y, cb, cr = ycbcr_px
            # OpenCV H <= 17 converted to Pillow H <= ~24
            hsv_match = h <= 24 and 15 <= s <= 170
            ycbcr_match = 135 <= cr <= 180 and 85 <= cb <= 135
            if hsv_match and ycbcr_match:
                skin_pixels += 1
        skin_proportion = skin_pixels / total_pixels

        if skin_proportion >= MIN_SKIN_PROPORTION:
            print(
                f"QC Pass (skin proportion: {skin_proportion:.2f}): {tile_image.name}"
            )
            qc.append(
                {
                    "tile_image": tile_image,
                    "result": "pass",
                    "reason": None,
                }
            )
        else:
            print(
                f"QC Fail (skin proportion: {skin_proportion:.2f}): {tile_image.name}"
            )
            qc.append(
                {
                    "tile_image": tile_image,
                    "result": "reject",
                    "reason": f"skin proportion too low: {skin_proportion:.2f}",
                }
            )

    return qc


def display_qc_results(qc_results, patient_folder_name):
    n_total = len(qc_results)
    n_pass = sum(1 for result in qc_results if result["result"] == "pass")
    n_pass_pc = (n_pass / n_total) * 100 if n_total > 0 else 0
    n_reject = sum(1 for result in qc_results if result["result"] == "reject")
    n_reject_pc = (n_reject / n_total) * 100 if n_total > 0 else 0
    print(
        f"QC results for {patient_folder_name}:\n\t{len(qc_results)} tiles checked\n\t-> {n_pass} passed ({n_pass_pc:.2f}%)\n\t-> {n_reject} rejected ({n_reject_pc:.2f}%)"
    )


def main():
    if not CLEANSED_DATA.exists():
        print(f"Error: CLEANSED_DATA folder does not exist: {CLEANSED_DATA}")
        return
    if not output_folder.exists():
        print(f"Output folder does not exist, creating: {output_folder}")
        output_folder.mkdir(parents=True, exist_ok=True)

    for patient_folder in [item for item in CLEANSED_DATA.iterdir() if item.is_dir()]:
        print(f"Processing patient folder: {patient_folder}")
        dst_pngs = output_folder / "1_converted_images" / patient_folder.name
        dst_pngs.mkdir(parents=True, exist_ok=True)

        dst_tiles = output_folder / "2_tiles" / patient_folder.name
        dst_tiles.mkdir(parents=True, exist_ok=True)

        baseline_folder = find_baseline_folder(patient_folder)
        if baseline_folder is None:
            print(f"Error: No imaging data found for patient folder: {patient_folder}")
            continue

        converted_images = convert_raw_images(baseline_folder, dst_pngs, use_wine=False)
        patient_tiles = create_tiles(converted_images, dst_tiles)
        qc_results = perform_tile_qc(patient_tiles)

        for qc_result in qc_results:
            tile_image = qc_result["tile_image"]
            tile_name = tile_image.name
            if qc_result["result"] == "reject":
                print(f"Moving rejected tile {tile_name} to QC_FAIL folder")
                (tile_image.parent / "QC_FAIL").mkdir(exist_ok=True)
                shutil.move(
                    tile_image,
                    Path(tile_image.parent, "QC_FAIL", tile_name),
                )
        print("Finished processing patient folder:", patient_folder.name)
        display_qc_results(qc_results, patient_folder.name)
    print("Done")


if __name__ == "__main__":
    main()
