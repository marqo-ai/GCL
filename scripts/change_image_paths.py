import os
import pandas as pd
import csv
import argparse


def update_csv_files(root_dir, img_dir):
    total_files = 0
    updated_files = 0

    # Normalize the image directory path
    img_dir = os.path.normpath(img_dir) + os.sep

    # Count total number of CSV files for progress indication
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                total_files += 1

    # Update CSV files
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(full_path)
                    if 'image_local' in df.columns:
                        df['image_local'] = df['image_local'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))
                        df.to_csv(full_path, index=False, quoting=csv.QUOTE_ALL)
                        updated_files += 1
                        print(f"Updated file ({updated_files}/{total_files}): {full_path}")
                    else:
                        print(f"No 'image_local' column in file: {full_path}")
                except Exception as e:
                    print(f"Error processing file {full_path}: {e}")

    print(f"Update completed. Total files updated: {updated_files}/{total_files}")


def main():
    parser = argparse.ArgumentParser(
        description="Update the 'image_local' column in CSV files to contain absolute paths.")
    parser.add_argument('root_directory', type=str, help='Root directory to search for CSV files.')
    parser.add_argument('image_directory', type=str,
                        help='Directory path to prepend to file names in the "image_local" column.')

    args = parser.parse_args()

    update_csv_files(args.root_directory, args.image_directory)


if __name__ == "__main__":
    main()
