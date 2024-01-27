import os
import marqo
import streamlit as st
from pipeline import Reach
from dataset_builder import GPTRequestHandler
from reusable_utils import extract_code, extract_content_from_gpt_response


os.environ["OPENAI_API_KEY"] = ""
dataset_description = None

uploads_dir = 'web_upload/datasets'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

with st.sidebar:
    uploaded_files = st.file_uploader('Choose a folder to upload', accept_multiple_files=True)

file_paths = []
for uploaded_file in uploaded_files:
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        content = uploaded_file.read()
        f.write(content)
    file_paths.append(os.path.abspath(file_path))


prompt = st.chat_input()
with st.status('Processing Input Params', expanded=True) as status:

    if prompt:
        flat_files_exist = any(f.endswith('.csv') for f in os.listdir('web_upload/datasets'))
        if flat_files_exist and not os.path.exists(os.path.join(uploads_dir, 'aggregated_data.csv')):
            st.write("Aggregating supplied data, this may take a few minutes.")
            handler = GPTRequestHandler()

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
            st.write('Aggregated set created successfully.')

        st.write('Beginning analysis...')
        r = Reach(
                marqo_client=marqo.Client(url="http://localhost:8882"),
                marqo_index='validation_testing', 
                train_set_path='aggregated_data.csv', 
                dataset_description=dataset_description, 
                goal_prompt=prompt,
                attempt_validation=True,
            )
            
        code_output, validated_code, so_what = r.main(n_suggestions=1, index_name=r.marqo_index)        
        with st.chat_message('user'):
            st.write(f'Result: {so_what}')
            # st.write(f'Analytics code: {validated_code}')
    
    status.update(label='All processes complete...', state='complete', expanded=True)