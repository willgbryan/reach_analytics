import os
import time
import marqo
import pandas as pd
from openai import OpenAI
import streamlit as st
from pipeline import Reach
from dataset_builder import GPTRequestHandler
from reusable_utils import (
    extract_code, 
    extract_content_from_gpt_response, 
    get_openai_client,
    clear_directory,
)


dataset_description = None

# Deployment config
base_dir = os.path.dirname('web_upload')
uploads_dir = os.path.join(base_dir, 'web_upload', 'datasets')
plots_dir = os.path.join(base_dir, 'web_upload', 'plots')
working_dir = os.path.join(base_dir, 'web_upload', 'working_dir')

os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(working_dir, exist_ok=True)
local = False

# Local config
uploads_dir = 'web_upload/datasets'
plots_dir = 'web_upload/plots'
working_dir = 'web_upload/working_dir'

if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

with st.sidebar:
    key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        client = get_openai_client(api_key=key)
    st.markdown("[Synthetic Datasets](https://github.com/willgbryan/reach_analytics/tree/main/synthetic_sets)")
    uploaded_files = st.file_uploader('Choose flat files to upload (.csv)', accept_multiple_files=True)
    if os.path.exists(os.path.join(uploads_dir, 'aggregated_data.csv')):
        df_aggregated = pd.read_csv(os.path.join(uploads_dir, 'aggregated_data.csv'))
        st.title("Aggregated Data")
        st.dataframe(df_aggregated)
    

file_paths = []
for uploaded_file in uploaded_files:
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        content = uploaded_file.read()
        f.write(content)
    file_paths.append(os.path.abspath(file_path))


ascii_text = """
`   ____                  __       ___                __      __  _          
   / __ \___  ____  _____/ /_     /   |  ____  ____ _/ /_  __/ /_(_)_________
  / /_/ / _ \/ __ `/ ___/ __ \   / /| | / __ \/ __ `/ / / / / __/ / ___/ ___/
 / _, _/  __/ /_/ / /__/ / / /  / ___ |/ / / / /_/ / / /_/ / /_/ / /__(__  ) 
/_/ |_|\___/\__,_/\___/_/ /_/  /_/  |_/_/ /_/\__,_/_/\__, /\__/_/\___/____/  
                                                    /____/    
"""

st.text(ascii_text)
prompt = st.chat_input("Lets extend your reach")
with st.status("Writing some code...", expanded=True) as status:
    if prompt:
        st.write(f'Prompt: {prompt}')
        # reset plots
        clear_directory(plots_dir)
        flat_files_exist = any(f.endswith('.csv') for f in os.listdir('web_upload/datasets'))
        if flat_files_exist and not os.path.exists(os.path.join(uploads_dir, 'aggregated_data.csv')):
            st.write("Aggregating supplied data, this may take a few minutes.")
            # reset session state to new
            clear_directory(working_dir)
            handler = GPTRequestHandler(client)

            response, supplied_file_paths, generated_df_summaries = handler.handle_files_and_send_request(
                file_paths=file_paths,
                prompt="Aggregate these datasets",
            )
            extracted_response = extract_content_from_gpt_response(response)
            data_eng_code = extract_code(extracted_response)

            validated_code = handler.code_validation_agent(
                code_to_validate=data_eng_code,
                file_paths=supplied_file_paths,
                context=[{"role": "user", "content": f"Dataframe Summaries: {generated_df_summaries}"}]
            )
            set = True
            st.write("Beginning analysis...")

        else:
            st.write('Existing aggregated set found, Beginning analysis...')

        r = Reach(
                local=local,
                client=client,        
                marqo_client=marqo.Client(url="http://localhost:8882"),
                marqo_index='validation_testing', 
                train_set_path='web_upload/datasets/aggregated_data.csv', 
                dataset_description=dataset_description, 
                goal_prompt=prompt,
                attempt_validation=True,
            )
            
        code_output, validated_code, so_what = r.main(n_suggestions=1, index_name=r.marqo_index)
        st.write('Analysis complete...')

        
        with st.chat_message('user'):
            st.write(f'Result: {so_what}')
            if os.path.exists(plots_dir):
                plot_files = os.listdir(plots_dir)
                
                for plot_file in plot_files:
                    if plot_file.endswith('.png'):
                        file_path = os.path.join(plots_dir, plot_file)
                        
                        st.image(file_path, caption=plot_file, use_column_width=True)
            else:
                st.error(f"The directory {plots_dir} does not exist.")
            st.code(validated_code)
    status.update(label="System Idle...", state="complete", expanded=True)