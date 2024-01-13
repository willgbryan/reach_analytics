import sys
import time
import tkinter as tk
from tkinter import filedialog
import marqo
from pipeline import Reach
import pandas as pd
import os
from openai import OpenAI
from dataset_builder import GPTRequestHandler
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask import send_from_directory
from reusable_utils import extract_code, extract_content_from_gpt_response


ui_mode = True if 'ui' in sys.argv else False

client = OpenAI(api_key='redact')

dataset_description = None

app = Flask(__name__)
CORS(app)
upload_folder = 'web_upload/datasets'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), upload_folder)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if ui_mode:

    @app.route('/datasets/<path:filename>', methods=['GET'])
    def serve_file_in_dir(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/upload_files', methods=['POST'])
    def dataset_handling():
        if 'file' not in request.files:
            return 'No file part', 400

        files = request.files.getlist('file')
        file_paths = []
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)

        try:
            if not file_paths or all(os.path.getsize(fp) == 0 for fp in file_paths):
                return jsonify({'message': 'No files or empty files uploaded'}), 400

            handler = GPTRequestHandler()
            print("Aggregating supplied data, this may take a few minutes.")

            response, supplied_file_paths, generated_df_summaries = handler.handle_files_and_send_request(
                file_paths=file_paths,
                prompt="Aggregate these datasets",
            )
            print("Response received, validating logic.")
            extracted_response = extract_content_from_gpt_response(response)
            data_eng_code = extract_code(extracted_response)

            validated_code = handler.code_validation_agent(
                code_to_validate=data_eng_code,
                file_paths=supplied_file_paths,
                context=[{"role": "user", "content": f"Dataframe Summaries: {generated_df_summaries}"}]
            )

            return jsonify({'message': 'Files processed successfully'})

        except pd.errors.EmptyDataError:
            print("Re-attempting aggregation, first pass failed.")
            return jsonify({'message': 'Error processing files'}), 500

    @app.route('/process_prompt', methods=['POST'])
    def process_prompt():
        data = request.json
        prompt = data['prompt']

        r = Reach(
            marqo_client=marqo.Client(url="http://localhost:8882"),
            marqo_index='validation_testing', 
            train_set_path='web_upload/datasets/aggregated_data.csv', 
            dataset_description=dataset_description, 
            goal_prompt=prompt,
            attempt_validation=True,
        )
        
        code_output, validated_code = r.main(n_suggestions=1, index_name=r.marqo_index)

        return jsonify({'codeOutput': code_output, 'validatedCode': validated_code})

def terminal_file_upload():
    print('Opening root...')
    root = tk.Tk()
    root.withdraw()
    print('Root opened, requesting explorer window...')
    time.sleep(0.1)
    file_paths_tuple = filedialog.askopenfilenames()
    root.update()
    file_paths = list(file_paths_tuple)
    if file_paths:
        dataset_handling_terminal(file_paths)
    print('Destroying root...')
    root.destroy()

def dataset_handling_terminal(file_paths):
    upload_folder = app.config['UPLOAD_FOLDER']

    try:
        if not file_paths or all(os.path.getsize(fp) == 0 for fp in file_paths):
            print('No files or empty files uploaded')
            return

        handler = GPTRequestHandler()
        print("Aggregating supplied data, this may take a few minutes.")

        response, supplied_file_paths, generated_df_summaries = handler.handle_files_and_send_request(
            file_paths=file_paths,
            prompt="Aggregate these datasets",
        )
        print("Response received, validating logic.")
        extracted_response = extract_content_from_gpt_response(response)
        data_eng_code = extract_code(extracted_response)

        validated_code = handler.code_validation_agent(
            code_to_validate=data_eng_code,
            file_paths=supplied_file_paths,
            context=[{"role": "user", "content": f"Dataframe Summaries: {generated_df_summaries}"}]
        )

        print(f'Validated code: {validated_code}')
        print('Files processed successfully')

    except pd.errors.EmptyDataError:
        print("Re-attempting aggregation, first pass failed.")

def process_prompt_terminal():
    prompt = input("Enter your prompt: ")

    r = Reach(
        marqo_client=marqo.Client(url="http://localhost:8882"),
        marqo_index='validation_testing',
        train_set_path='aggregated_data.csv',
        dataset_description=dataset_description,
        goal_prompt=prompt,
        attempt_validation=True,
    )

    code_output, validated_code = r.main(n_suggestions=1, index_name=r.marqo_index)

    print(f"Code Output:\n{code_output}\n")
    print(f"Validated Code:\n{validated_code}\n")

def clear_data_files():
    try:
        os.remove('aggregated_data.csv')
        print('aggregated_data.csv has been removed.')
    except FileNotFoundError:
        print('aggregated_data.csv does not exist or has already been removed.')

    try:
        os.remove('data_context.txt')
        print('data_context.txt has been removed.')
    except FileNotFoundError:
        print('data_context.txt does not exist or has already been removed.')
    
    try:
        os.remove('memory.txt')
        print('memory.txt has been removed.')
    except FileNotFoundError:
        print('memory.txt does not exist or has already been removed.')

if __name__ == '__main__':
    ascii_art = "\033[38;2;199;254;0m" + """
    ____                  __       ___                __      __  _          
   / __ \___  ____  _____/ /_     /   |  ____  ____ _/ /_  __/ /_(_)_________
  / /_/ / _ \/ __ `/ ___/ __ \   / /| | / __ \/ __ `/ / / / / __/ / ___/ ___/
 / _, _/  __/ /_/ / /__/ / / /  / ___ |/ / / / /_/ / / /_/ / /_/ / /__(__  ) 
/_/ |_|\___/\__,_/\___/_/ /_/  /_/  |_/_/ /_/\__,_/_/\__, /\__/_/\___/____/  
                                                    /____/    
    """ + "\033[0m"
    if ui_mode:
        app.run(debug=True)
    else:
        print(ascii_art)
        while True:
            command = input("Enter command (type 'exit' to quit): ")
            if command == 'exit':
                print('Tearing down...')
                break
            elif command == 'file-upload':
                terminal_file_upload()
            elif command == 'process-prompt':
                process_prompt_terminal()
            elif command == 'clear-data':
                clear_data_files()