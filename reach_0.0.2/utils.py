import os
import traceback
import warnings
import pandas as pd
import numpy as np
import openai
from typing import Any, List, Dict, Union

# # TODO code validation agent implementation requires a refactor to support openai >= 1.0.0
# def extract_content_from_gpt_response(function):
#     def wrapper(*args, **kwargs):
#         response_body = function(*args, **kwargs)
#         try:
#             return response_body.choices[0].message.content
#         except AttributeError:
#             return None
#     return wrapper

# @extract_content_from_gpt_response
# def send_request_to_gpt(
#         role_preprompt: str, 
#         prompt: str,
#         context: Union[str, Dict[str, str]],  
#         stream: bool = False
#         ) -> str:
    
#     if isinstance(context, str):
#         context = [{"role": "user", "content": context}]

#     response = client.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=[
#             {
#                 "role": "system",
#                 "content": role_preprompt,
#             },
#             *context,
#             {
#                 "role": "user",
#                 "content": prompt,
#             },
#         ],
#         stream=stream,
#     )

#     return response

def dataframe_summary( 
            df: pd.DataFrame, 
            dataset_description: str = None, 
            sample_rows: int = 5, # TODO Dynamically assign this value to a chunk representative of the set
            sample_columns: int = 14 # TODO Dynamically assign to the max number of columns
            ) -> str:
        """
        Create a text summary of a Pandas DataFrame.

        Parameters:
            df (Pandas DataFrame): The dataframe to be summarized.
            sample_rows (int): The number of rows to sample.
            sample_columns (int): The number of columns to sample.

        Returns:
            A markdown string with a text summary of the dataframe.
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

        # TODO represent categoricals in the return summary (categoricals flagging can often be a false positive)
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

def process_files(file_paths: list):
    """
    Process the uploaded files and preserve file paths.
    """
    summaries = []
    for file_path in file_paths:
        if file_path.endswith('.xlsx') or file_path.endswith('.csv'):
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            summary = dataframe_summary(df)
            summaries.append(summary)

    return {'dataframe_summaries': summaries, 'file_paths': file_paths}

# def code_validation_agent(
#             self, 
#             code_to_validate: str, 
#             context: List[Dict[str, str]], 
#             max_attempts: int = 10,
#         ) -> str:

#         warnings.filterwarnings("ignore")  # Ignore warnings
#         attempts = 0

#         while attempts < max_attempts:
#             print(f'Validation attempt: {attempts}')

#             try:
#                 exec(code_to_validate)
#                 print('\033[38;2;199;254;0m' + 'Code executed without errors...' + '\033[0m')

#                 # self.log.info("Code executed without errors.")
#                 return code_to_validate

#             except Exception as e:
#                 error_message = str(e)
#                 error_traceback = traceback.format_exc()
#                 print(error_traceback)
#                 print(error_message)
#                 # self.log.info(error_message)

#                 # Debugging via GPT-4
#                 response = send_request_to_gpt(
#                     role_preprompt=self.validation_preprompt,
#                     context=context,
#                     prompt=f"""
#                     Debug the following python code: {code_to_validate}. \n\nError:\n{error_message}\n\nTraceback:\n{error_traceback}\n\n.
#                     Training data can be found at {self.train_set_path} 
#                     You must return THE ENTIRE ORIGINAL CODE BLOCK WITH THE REQUIRED CHANGES.
#                     """,
#                 )