"""
Developed by:
Frazer Institute, The University of Queensland, Brisbane, Australia

If this software contributes to published research,
please acknowledge the above institution.
"""

from pathlib import Path
import pandas as pd

# Set these variables:

# If `True`, will scan entire export but only save data for first 5 JSON files (faster)
# Use this to check that the script works in your environment, and also to find any
# v1.0.9 JSON files that need to be updated.
test_mode = True

# 1. The root path of your cleansed exported data
input_cleansed_export = Path(r"c:/users/uqvectra/desktop/exports")

# 2. The destination folder where each participant's CSV file is saved
output_per_participant = Path(r"c:/users/uqvectra/desktop/participant_lesions")

# 3. The destination path for the output combined CSV file of filtered lesions
output_combined_lesions = Path(r"c:/users/uqvectra/desktop/filtered_lesions.csv")

# 4. The destination path for the baseline newest JSON list CSV file
output_baseline_json_list = Path(r"c:/users/uqvectra/desktop/baseline_json_list.csv")

# Do not change anything below this line
# -------------------------------------------------------

if not input_cleansed_export.exists():
    raise FileNotFoundError(
        f"Cleansed export path {input_cleansed_export} does not exist."
    )

if not output_per_participant.exists():
    raise FileNotFoundError(
        f"Output per participant folder {output_per_participant} does not exist."
    )

if output_combined_lesions.exists():
    raise FileExistsError(
        f"Output CSV path {output_combined_lesions} already exists. Please choose a different path or delete the existing file."
    )

# Create temporary file to check write permissions
try:
    with open(output_combined_lesions, "w") as f:
        f.write("test")
    output_combined_lesions.unlink()  # Remove the test file
except Exception as e:
    raise PermissionError(
        f"Cannot write to output CSV path {output_combined_lesions}. Please check permissions. Error: {e}"
    )


# Function to check if the json file seems like a valid lesion data json
def is_valid_json(json_path, min_size_mb=1):
    json_path = Path(json_path)
    if (
        json_path.parent.name == "analysis"
        and json_path.parent.parent.parent.parent == input_cleansed_export
        and json_path.stat().st_size > min_size_mb * 1e6
        and json_path.name.startswith("lesion_data")
    ):
        return True
    return False


# Adds participant id, timestamp, json version to the file manifest
def add_manifest_cols(df):
    df["json_path"] = df["json_path"].map(Path)
    manifest = df["json_path"].apply(
        lambda p: pd.Series(
            {
                "participant": p.parent.parent.parent.name,
                "timestamp": p.parent.parent.name,
                "filename": p.stem,
                "json_version": p.stem.split("_")[-1],
            }
        )
    )
    return pd.concat([df, manifest], axis=1)


# Try to load the lesion data json with a lot of safety checks
def safe_load_json(json_path, participant, timestamp):
    json_path = Path(json_path)
    json_ver = json_path.stem.split("_")[-1]
    if json_ver == "1.0.9":
        print(
            f"* Warning: Could not use JSON from following record due to old JSON version (1.0.9). Please update using Vectra software:\n\tGUID: {participant}, timestamp: {timestamp}"
        )
        return None
    ld = pd.read_json(json_path)
    # check if json has ["root"]["children"] key
    if "root" in ld and "children" in ld["root"]:
        lesions = pd.DataFrame(ld["root"]["children"])
    else:
        print(
            f"Warning: JSON at {json_path} for participant {participant} at timestamp {timestamp} is missing 'root' or 'children' keys."
        )
        return None

    # check if lesions dataframe has columns: nevi_confidence, majorAxisMM
    # these are missing in v1.0.9
    required_cols = ["nevi_confidence", "majorAxisMM"]
    if all(col in lesions.columns for col in required_cols):
        lesions = lesions.copy()
        lesions = lesions.loc[:, ~lesions.columns.str.startswith("img")]
        if "boundary" in lesions.columns:
            lesions = lesions.drop(columns=["boundary"])
        # lesions = lesions.loc[
        #     (lesions["nevi_confidence"] >= 80) & (lesions["majorAxisMM"] >= 2)
        # ].copy()
        lesions["src_file"] = str(json_path)
        lesions["participant_guid"] = participant
        lesions["timestamp"] = timestamp
        return lesions
    else:
        print(
            f"Warning: JSON at {json_path} for participant {participant} at timestamp {timestamp} is missing required columns in 'children'."
        )
        return None


if __name__ == "__main__":

    # Searches the cleansed export for valid json files
    print("* Searching for for all valid JSON files in the cleansed export path...")
    valid_json_files = [
        f for f in input_cleansed_export.rglob("*.json") if is_valid_json(f)
    ]

    if len(valid_json_files) == 0:
        print("No valid JSON files found in the cleansed export path.")
        exit()
    else:
        pt_count = set([x.parent.parent.parent.name for x in valid_json_files])
        print(pt_count)
        print(
            f"Found:\n\t- {len(pt_count)} participants\n\t- {len(valid_json_files)} valid JSON files across all visits"
        )
        print("Determining newest JSON version available for each baseline visit...")

    # Put the list of valid json files into a dataframe
    all_jsons_df = pd.DataFrame(valid_json_files, columns=["json_path"])
    all_jsons_df = add_manifest_cols(all_jsons_df)

    # Find the newest json (by version) per baseline timestamp for each participant
    # JSON version is determined by filename, in the format `lesion_data_X.Y.Z.json`
    baseline_newest_jsons_df = (
        all_jsons_df.sort_values(
            by=["participant", "timestamp", "filename"],
            ascending=[True, True, False],  # earliest timestamp, max filename
        )
        .drop_duplicates(subset="participant", keep="first")
        .reset_index(drop=True)
    )

    print("Saving the baseline newest JSON list to a CSV file...")
    baseline_newest_jsons_df.to_csv(output_baseline_json_list, index=False)

    if test_mode:
        print(
            "Test mode is ON. Only processing the first 3 JSON files.\nSet test_mode = False to process all files."
        )
        baseline_newest_jsons_df = baseline_newest_jsons_df.head(3)

    # Get the list of valid baseline json files and load them safely
    output_dfs = []
    for idx, row in baseline_newest_jsons_df.iterrows():  # type: ignore
        json_data = safe_load_json(
            row["json_path"], row["participant"], row["timestamp"]
        )
        if json_data is not None:
            dst_fname = f"{row['participant']}_{row['timestamp']}_lesions.csv"
            dst_path = output_per_participant / dst_fname
            print(
                f"Saving CSV file for {row['participant']} at timestamp {row['timestamp']} to {dst_path}..."
            )
            json_data.to_csv(dst_path, index=False)
            output_dfs.append(json_data)

    # Combine all the loaded lesion data into a single dataframe
    # Save the combined dataframe to a CSV file
    if output_dfs:
        print(
            f"Combining data from {len(output_dfs)} participants into a single CSV file ({output_combined_lesions})..."
        )
        output_df = pd.concat(output_dfs, ignore_index=True)
        output_df.to_csv(output_combined_lesions, index=False)
    else:
        print(
            "No valid JSON files found or all JSON files were invalid. No output CSV generated."
        )

    print("Script execution complete.")
