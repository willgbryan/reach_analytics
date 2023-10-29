import os
import ast
import json

def write_json_to_file(filename, data):
    """
    Write a JSON dictionary to a text file.

    Args:
        filename (str): The name of the file to create and write to.
        data (dict): The JSON data to be written to the file.
    """
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def read_json_from_file(filename):
    """
    Read a JSON file and parse its contents into a dictionary.

    Args:
        filename (str): The name of the JSON file to read.

    Returns:
        dict: A dictionary containing the parsed JSON data.
        None: If the file does not exist or cannot be parsed as JSON.
    """
    if not os.path.exists(filename):
        print(f"The file '{filename}' does not exist.")
        return None

    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in '{filename}': {str(e)}")
        return None

def append_data_to_file(filename, data):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = json.load(f)
    else:
        content = []

    highest_step = max([entry.get("step", 0) for entry in content], default=0)

    data["step"] = highest_step + 1

    content.append(data)

    with open(filename, 'w') as f:
        json.dump(content, f)

def store_data_context(preprocessing_context: str, df_context: str):
    file_path = "data_context.txt"
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as _:
            pass

    data_to_write = {
        "preprocessing_context": preprocessing_context,
        "dataframe_summary": df_context
    }

    data_str = str(data_to_write)

    with open(file_path, "w") as file:
        file.write(data_str)

def load_data_context():
    with open("data_context.txt", "r") as file:
        content = file.read()
        
    data_dict = ast.literal_eval(content)
    
    df_context = data_dict["dataframe_summary"]
    preprocessing_context = data_dict["preprocessing_context"]
    
    return preprocessing_context, df_context