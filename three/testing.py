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

client = OpenAI(api_key='redact')

dataset_description = None

app = Flask(__name__)
CORS(app)
upload_folder = 'web_upload/datasets'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), upload_folder)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

    # Now that you have file_paths, you can process them
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
        train_set_path='C:/Users/willb/OneDrive/Documents/GitHub/placeholder1/web_upload/datasets/aggregated_data.csv', 
        dataset_description=dataset_description, 
        goal_prompt=prompt,
        attempt_validation=True,
    )
    
    code_output, validated_code = r.main(n_suggestions=1, index_name=r.marqo_index)

    return jsonify({'codeOutput': code_output, 'validatedCode': validated_code})

if __name__ == '__main__':
    app.run(debug=True)