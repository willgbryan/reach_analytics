import pandas as pd
import os

def dict_to_dataframe_and_save(data_dict, file_path):
    """
    Converts a dictionary to a DataFrame and saves or appends it to a CSV file.

    Parameters:
    data_dict (dict): The dictionary to convert.
    file_path (str): Path to the CSV file.

    Returns:
    None
    """
    df_new = pd.DataFrame.from_dict(data_dict)

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        df_combined.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")

