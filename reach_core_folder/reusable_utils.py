import os
import openai
import shutil
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Any

def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

def dict_to_dataframe(data_dict: Dict, file_path: str):
    """
    Converts a dictionary to a DataFrame and saves or appends it to a CSV file.

    Parameters:
    data_dict (dict): The dictionary to convert.
    file_path (str): Path to the CSV file.

    Returns:
    None
    """
    df_new = pd.DataFrame.from_dict([data_dict])

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        df_combined.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")

def extract_code(message: str) -> str:
        substr = message.find('```python')
        incomplete_code = message[substr + 9 : len(message)]
        substr = incomplete_code.find('```')
        code = incomplete_code[0:substr]
        return code

def send_request_to_gpt(
            client,
            role_preprompt: str, 
            prompt: str,
            context: Dict[str, str],  
            stream: bool = False
            ) -> str:

        # Handle string input for context
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                # Establish the context of the conversation
                {
                    "role": "system",
                    "content": role_preprompt,
                },
                # Previous interactions
                *context,
                # The user's code or request
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=stream,
        )
        return response

def dataframe_summary( 
            df: pd.DataFrame, 
            dataset_description: str = None, 
            sample_rows: int = 5, 
            sample_columns: int = 14
            ) -> str:
        """
        Create a GPT-friendly summary of a Pandas DataFrame.

        Parameters:
            df (Pandas DataFrame): The dataframe to be summarized.
            sample_rows (int): The number of rows to sample.
            sample_columns (int): The number of columns to sample.

        Returns:
            A markdown string with a GPT-friendly summary of the dataframe.
        """
        num_rows, num_cols = df.shape

        # Column Summary
        missing_values = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
        missing_values['% Missing'] = missing_values['Missing Values'] / num_rows * 100
        missing_values = missing_values.sort_values(by='% Missing', ascending=False).head(5)

        # Basic data typing support could go here but it may not be necessary

        # Basic summary statistics for numerical and categorical columns
        numerical_summary = df.describe(include=[np.number])
        
        has_categoricals = any(df.select_dtypes(include=['category', 'datetime', 'timedelta']).columns)

        if has_categoricals:
            categorical_summary = df.describe(include=['category', 'datetime', 'timedelta'])
        else:
            categorical_summary = pd.DataFrame(columns=df.columns)

        sampled = df.sample(min(sample_columns, df.shape[1]), axis=1).sample(min(sample_rows, df.shape[0]), axis=0)

        # Constructing a GPT-friendly output:
        if dataset_description is not None:
            output = (
                f"Here's a summary of the dataframe:\n"
                f"- Rows: {num_rows:,}\n"
                f"- Columns: {num_cols:,}\n"
                f"- All columns: {df.columns:,}\n\n"

                f"Column names and their descriptions:\n"
                f"{dataset_description}"

                f"Top columns with missing values:\n"
                f"{missing_values.to_string()}\n\n"

                f"Numerical summary:\n"
                f"{numerical_summary.to_string()}\n\n"

                f"A sample of the data ({sample_rows}x{sample_columns}):\n"
                f"{sampled.to_string()}"
            )
        
        else: 
            output = (
                f"Here's a summary of the dataframe:\n"
                f"- Rows: {num_rows:,}\n"
                f"- Columns: {num_cols:,}\n\n"

                f"Top columns with missing values:\n"
                f"{missing_values.to_string()}\n\n"

                f"Numerical summary:\n"
                f"{numerical_summary.to_string()}\n\n"

                f"A sample of the data ({sample_rows}x{sample_columns}):\n"
                f"{sampled.to_string()}"
            )

        return output

def extract_content_from_gpt_response(response: Any) -> str:
    """
    Extracts content from a GPT response.

    Parameters:
    response (Any): The GPT response to extract content from.

    Returns:
    str: Extracted content.
    """
    return response.choices[0].message.content

def clear_directory(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)