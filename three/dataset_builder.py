import tkinter as tk
from tkinter import filedialog
import traceback
import warnings
import openai
import pandas as pd
import numpy as np
from typing import List, Any, Dict
from reusable_utils import (
    dataframe_summary,
    dict_to_dataframe,
    send_request_to_gpt,
    extract_code,
    extract_content_from_gpt_response
)

from log_module import logger
from flask import Flask, app, request, jsonify
from flask_cors import CORS
import os

class GPTRequestHandler:

    def get_data_engineer_preprompt(self, file_paths: list):
        """
        Generates a dynamic preprompt for a data engineer with the file paths.
        """
        file_paths_str = ", ".join(file_paths)
        return f"""
            You are a professional data engineer and your role is to find ways to aggregate disparate datasets using python code.
            You will be provided with summary information about the incoming data including available columns.
            The summary information can be found in the context at "Dataframe Summaries".
            Feature engineering and other similar techniques can be useful in accomplishing your task.
            If there are no like keys to join on, you must create new columns or make assumptions to create joins.
            Using 'Unnamed: X' is not allowed as a column name.
            The output .csv must be titled 'aggregated_data.csv'
            Data can be found at {file_paths_str}.
            The final output of the code should be all data in an aggregated dataset written to a csv.
            Format all code in a single block like below:
            ```python
            # code
            aggregated_data.to_csv('aggregated_data.csv')
            ```
        """.strip()

    def process_files(self, file_paths: list):
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

    def code_validation_agent(
        self, 
        code_to_validate: str, 
        context: List[Dict[str, str]], 
        max_attempts: int = 10,
        file_paths: List[str] = None,
        web: bool = False,
    ) -> str:

        warnings.filterwarnings("ignore")  # Ignore warnings
        attempts = 0

        while attempts < max_attempts:
            try:
                exec(code_to_validate)
                print("Code executed without errors.")
                # self.log.info("Code executed without errors.")
                return code_to_validate

            except Exception as e:
                error_message = str(e)
                error_traceback = traceback.format_exc()
                print(error_message)
                # self.log.info(error_message)

                file_paths = file_paths
                file_paths_str = ", ".join(file_paths)


                # Debugging via GPT-4
                response = send_request_to_gpt(
                    role_preprompt=f"""
                    As a python coding assistant, your task is to help users debug the supplied code using the context, code, and traceback provided.
                    Simply return the remedied code, but try to be proactive in debugging. If you see multiple errors that can be corrected, fix them all.
                    Data can be found at {file_paths_str}.
                    You must return THE ENTIRE ORIGINAL CODE BLOCK WITH THE REQUIRED CHANGES.
                    Format your response as:

                    ```python
                    # code
                    ```
                    """.strip(),
                    context=context,
                    prompt=f"""
                    Debug the following python code: {code_to_validate}. \n\nError:\n{error_message}\n\nTraceback:\n{error_traceback}\n\n.
                    Training data can be found at {file_paths_str} 
                    You must return THE ENTIRE ORIGINAL CODE BLOCK WITH THE REQUIRED CHANGES.
                    """,
                    stream=False
                )

                suggestion = extract_code(
                    (extract_content_from_gpt_response(
                        response
                        )
                    )
                )

                print(f"Updated Code: \n{suggestion}")
                # self.log.info(f"Updated Code: \n{suggestion}")

                dict_to_dataframe(
                        data_dict = {
                            'code_to_validate': code_to_validate,
                            'error_message': error_message,
                            'traceback': traceback,
                            'updated_code': suggestion,
                            },
                        file_path = 'dataset_builder_validation_finetuning_set.csv',
                    )

                code_to_validate = suggestion
                attempts += 1

        print("Max attempts reached. Code is still broken.")
        # self.log.info("Max attempts reached. Code is still broken.")
        return None

    def handle_files_and_send_request(
        self, 
        prompt: str, 
        file_paths: list,
        stream: bool = False,
    ):
        file_processing_result = self.process_files(file_paths)
        file_paths = file_processing_result['file_paths']
        summary_dict = file_processing_result['dataframe_summaries']

        role_preprompt = self.get_data_engineer_preprompt(file_paths)

        code = send_request_to_gpt(
            role_preprompt=role_preprompt,
            prompt=prompt,
            context=[{"role": "user", "content": f"Dataframe Summaries: {summary_dict}"}],
            stream=stream,
        )

        return code, file_paths, summary_dict

