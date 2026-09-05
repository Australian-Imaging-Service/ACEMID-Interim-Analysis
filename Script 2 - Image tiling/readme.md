# Interim Study - Script 2

Document prepared by [Adam Mothershaw](mailto:a.mothershaw@uq.edu.au)

## Summary

This Python script scans the cleansed exported data for each participant's baseline folder, then converts the CR2 images in that folder to PNG and corrects their rotation. Finally, it splits each PNG into 5x9 tiles and checks each tile for skin content and against a reject list. Tiles that fail the checks are moved to a seperate folder. This process is performed in order to prepare tiles for running through the photodamage AI model.

## Requirements

* Windows, macOS or Linux
  * If using macOS or Linux, must have [wine](https://www.winehq.org) intalled and set the `use_wine` parameter in the `convert_raw_images` function call - see [Instructions](#instructions) step 4.
* `script_2.py` file
* **Pillow** Python package: `python -m pip install Pillow`
* Cleansed export of interim study participants
* `raw_converter_v1` folder containing:
  * `convertCR2.exe`, `query3d.dll`, `tbb64.dll`, `tbbmalloc_proxy.dll`, `tbbmalloc.dll`
* `raw_converter_v2` folder containing:
  * `WbRawToPng.exe`, `tbb64.dll`
* Time - expect this script to process around 15 to 18 participants per hour (3.5 to 4 mins per participant). The image format conversion (CR2 -> PNG) and tiling steps are multi-threaded (up to 8 threads, can be adjusted using the `MAX_THREADS` variable on line 10). The tile QC step is single-threaded as it seemed like multi-threading didn't meaningfully speed it up. The QC step is the slowest part of the operation.

## Notes

* The script will somewhat handle being interrupted or terminated and then re-run. It will not re-convert CR2s that have already been converted - but it will re-perform the tiling and QC. The tiling is very quick, but re-doing the QC will be slow. If you need to terminate the script and re-run at a later date, move the already-tiled/QC'd participants out of the input folder, to prevent them being re-processed.

## Inputs

* Cleansed export of interim study participants

## Outputs

* `1_converted_images` folder containing per-participant subfolders. Each subfolder has 46 PNG images, converted from the original Canon CR2 images.
* `2_tiles` folder containing per-participant subfolders. Each subfolder contains tiles named as `<guid>_<timestamp>_<camera_id>_<tile_id>.png`. Tiles directly within the participant subfolder have passed QC, tiles that have failed QC will be placed within a folder named `QC_FAIL`.

## Instructions

1. Open `script_2.py` in a text editor/IDE.
2. Edit the `CLEANSED_DATA` variable (line 8) to point to the path of your cleansed data. The script will check that the path exists, and exit if the path is not found.
3. Edit the `output_folder` variable (line 9) to point to the path where you want all the converted images and tiles to be saved. The script will check if the path exists, and create the path if it does not exist.
4. Optional: edit the `MAX_THREADS` variable (line 10), depending on the CPU on which the script is being run. If unsure, leave as-is. Setting too high may slow down the process.
5. Change the `use_wine` parameter (line 328) from `False` to `True` if you are using macOS or Linux.
6. Save your changes to the file `script_2.py`.
7. Run the script from the command-line: `python script_2.py`
8. Optional: perform a quick manual visual scan of tiles to ensure that tiles have been QC'd correctly.
9. The next step is to run the QC-passing tiles through the photodamage AI prepared by Zhen Yu.