import tkinter as tk
from tkinter import filedialog
import openai
import pandas as pd
import numpy as np
from typing import List, Any, Dict
from reusable_utils import (
    dataframe_summary,
    send_request_to_gpt,
    extract_code
)

class GPTRequestHandler:
    def __init__(self):
        pass

    def upload_files(self, web: bool = False):
        """
        File upload function for both local and web environments.
        Set `web` to True when using in a web environment with Flask.
        """
        file_paths = []

        if web:
            from flask import request
            if request.method == 'POST':
                files = request.files.getlist('file')
                file_paths = [f.filename for f in files]  # Example: just getting file names
        else:
            # Local environment using tkinter
            root = tk.Tk()
            root.withdraw()
            file_paths = filedialog.askopenfilenames()

        self.dataset_path = ", ".join(file_paths)

        return file_paths
    
    def get_data_engineer_preprompt(self, file_paths: str):
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

            Data can be found at {file_paths_str}.
            
            Format your response as:

            ```python
            # code
            ```

            The final output of the code should be an aggregated dataset written to a csv.
            """.strip()

    def process_files(self, files):
        """
        Process the uploaded files and preserve file paths.
        """
        summaries = []
        file_paths = []  # List to store file paths

        for file_path in files:
            file_paths.append(file_path)  # Store the file path

            if file_path.endswith('.xlsx') or file_path.endswith('.csv'):
                df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
                summary = dataframe_summary(df)
                summaries.append(summary)

        return {'dataframe_summaries': summaries, 'file_paths': file_paths}

    def handle_files_and_send_request(
            self, 
            prompt: str, 
            stream: bool = False, 
            web: bool = False,
        ):
        file_processing_result = self.process_files(
            self.upload_files(web=web)
        )
        
        file_paths = file_processing_result['file_paths']
        summary_dict = file_processing_result['dataframe_summaries']

        role_preprompt = self.get_data_engineer_preprompt(file_paths)

        print(f"Preprompt: {role_preprompt}")
        
        code = send_request_to_gpt(
            role_preprompt=role_preprompt,
            prompt=prompt,
            context=[{"role": "user", "content": f"Dataframe Summaries: {summary_dict}"}],
            stream=stream,
        )
        
        return code, file_paths



openai.api_key = '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'
handler = GPTRequestHandler()

response = handler.handle_files_and_send_request(
    prompt="Aggregate these datasets"
)

print(extract_code(response))

