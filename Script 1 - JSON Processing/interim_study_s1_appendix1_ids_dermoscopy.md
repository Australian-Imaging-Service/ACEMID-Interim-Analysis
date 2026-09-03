# Interim Study - Script 1 Appendix 1 - Lesion IDs and dermoscopy
Adam Mothershaw <amothershaw@uq.edu.au>

## Background
Every lesion detected by the Vectra software has two identifiers: the UUID stored within the JSON files, and the Lesion ID presented to the end-user. UUIDs are long unique strings, e.g. `123e4567-e89b-12d3-a456-426614174000`, while the Lesion ID is an integer such as `1`, `2`, `3`. In REDCap, lesions of interest are only recorded by their Lesion ID. In order to find these lesions in the JSON file, we must generate spreadsheets that link the UUID and Lesion ID.

## Method Overview

We will use the `VectraDBTool.exe` utility to export "lesion analysis data" files and a "dermoscopy data" file.

Lesion analysis data files are CSV files (one file per participant) that contain one row per tagged lesion per visit - and importantly, have columns for both the UUID and Lesion ID. This will allow the Lesion ID column to be joined to the exported JSON data.

The dermoscopy data file contain dermoscopy metadata, including the Lesion ID, capture date, and capture device for all participants.

## Steps

1. Locate the Vectra application file **vectra.exe** on the PC. Often it's at: **C:\vectra\bin\vectra.exe**
2. Within the same `bin` folder, find and run the file **VectraDBTool_v1.8.exe** that was copied during the initial data export [(file and instructions available here)](https://drive.google.com/drive/folders/1LUM5S_r9DIIUjsDktOAR2T0HFq8IMtHh?role=reader).
3. When the **VectraDBTool** opens, you will see a small popup window titled Database list. Select the ACEMID database from the **Select a database:** drop-down list and click OK.
4. Click the **DermX Utils** menu heading, and then select **Export lesion analysis data**. 
5. Under **Export Directory**, select the folder to where the CSV files will be saved (one file per participant, approx 50KB-100KB per file).
6. Ensure that the **Export manually tagged lesions only** option is checked ✅, then click OK.
7. A window will popup with a textbox labelled **Please enter MRNs to process**. Into this textbox, paste the list of interim study participant IDs relevant to the current database. [Study ID lists are available here](https://drive.google.com/drive/folders/1jptJUiNOq493UbAKL1hRlKgleKAOInNE?role=reader).
8. Click the **Process** button and wait for the file processing to complete. Expect it to take 20-30 seconds per participant. The **VectraDBTool** may appear to stop responding during the export, simply wait for the process to complete and do not prematurely terminate the process. You may check the progress by looking at the number of CSV files created in the export directory selected in Step 5.
9. When the process has finished, click the **DermX Utils** menu heading in the main **VectraDBTool** window, then select **Export dermoscopy data**. 
10. Under **Export Directory**, select the folder to where the CSV file (and optionally, dermoscopy images - see Step 11) will be saved.
11. **NOTICE**: at this point, you can optionally export dermoscopy images in addition to the dermoscopy metadata. Exporting the images is recommended as they will likely be used for other upcoming studies, however it will increase both the processing time and storage requirements for this step. With that in mind: choose whether to tick or untick the **Export dermoscopy images** option. Untick the **Export dexi data** option. Click OK.
12. Similar to Step 7, a window will popup with a textbox labelled **Please enter MRNs to process**. Paste the same list of study IDs used in Step 7, and click **Process**.
13. As before, the **VectraDBTool** may appear to stop responding while the data is being processed. Do not prematurely terminate the program. Expect this to take approx. 10-20 seconds per participant if dermoscopy images are not being exported, or up to 1-2 mins per participant if dermoscopy images are being exported (this depends on how many dermoscopy images the participant has).