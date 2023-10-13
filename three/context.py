import os
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

