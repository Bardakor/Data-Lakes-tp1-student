import os
import pandas as pd

def unpack_data(input_dir, output_file):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """
    # Step 1: Initialize an empty list to store DataFrames
    dataframes = []

    # Step 2: Loop over files in the input directory
    for filename in sorted(os.listdir(input_dir)):
        # Step 3: Check if the file matches our data file pattern
        if filename.startswith('data-') and filename.endswith('-of-00010'):
            file_path = os.path.join(input_dir, filename)
            
            # Step 4: Read the CSV file using pandas
            df = pd.read_csv(file_path)
            
            # Step 5: Append the DataFrame to the list
            dataframes.append(df)
            print(f"Processed {filename}")

    # Step 6: Concatenate all DataFrames
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Step 7: Save the combined DataFrame to output_file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")
    else:
        print("No data files found to combine")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
