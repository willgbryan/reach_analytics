import openai
import numpy as np
import pandas as pd
import tiktoken
import json
import os
import sys
from io import StringIO
from typing import Dict, Optional
import traceback

# langchain imports
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents.mrkl.base import ZeroShotAgent
from typing import Any, Dict, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.python.prompt import PREFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.tools.python.tool import PythonREPLTool

# load testing data
test_series_set = pd.read_parquet('cmi_sleep_states/test_series.parquet')
train_series_set = pd.read_parquet('C:/Users/willb/Downloads/train_series.parquet')
train_events_set = pd.read_csv('cmi_sleep_states/train_events.csv')

# expose the key alot
openai.api_key = 'redact'
os.environ["OPENAI_API_KEY"] = 'redact'

train_test_path = "C:/Users/willb/Downloads/train_series.parquet"

goal_prompt = "What kind of machine learning solution would you recommend for this dataset if I am looking to detect sleep onset and wake. You will develop a model trained on wrist-worn accelerometer data in order to determine a persoms sleep state."

attempt_validation = False

dataset_description = """Dataset Description
        The dataset comprises about 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep. Your task is to detect the occurrence of these two events in the accelerometer series.

        While sleep logbooks remain the gold-standard, when working with accelerometer data we refer to sleep as the longest single period of inactivity while the watch is being worn. For this data, we have guided raters with several concrete instructions:

        A single sleep period must be at least 30 minutes in length
        A single sleep period can be interrupted by bouts of activity that do not exceed 30 consecutive minutes
        No sleep windows can be detected unless the watch is deemed to be worn for the duration (elaborated upon, below)
        The longest sleep window during the night is the only one which is recorded
        If no valid sleep window is identifiable, neither an onset nor a wakeup event is recorded for that night.
        Sleep events do not need to straddle the day-line, and therefore there is no hard rule defining how many may occur within a given period. However, no more than one window should be assigned per night. For example, it is valid for an individual to have a sleep window from 01h00–06h00 and 19h00–23h30 in the same calendar day, though assigned to consecutive nights
        There are roughly as many nights recorded for a series as there are 24-hour periods in that series.
        Though each series is a continuous recording, there may be periods in the series when the accelerometer device was removed. These period are determined as those where suspiciously little variation in the accelerometer signals occur over an extended period of time, which is unrealistic for typical human participants. Events are not annotated for these periods, and you should attempt to refrain from making event predictions during these periods: an event prediction will be scored as false positive.

        Each data series represents this continuous (multi-day/event) recording for a unique experimental subject.

        Note that this is a Code Competition, in which the actual test set is hidden. In this public version, we give some sample data in the correct format to help you author your solutions. The full test set contains about 200 series.

        Files and Field Descriptions
        train_series.parquet - Series to be used as training data. Each series is a continuous recording of accelerometer data for a single subject spanning many days.
        series_id - Unique identifier for each accelerometer series.
        step - An integer timestep for each observation within a series.
        timestamp - A corresponding datetime with ISO 8601 format %Y-%m-%dT%H:%M:%S%z.
        anglez - As calculated and described by the GGIR package, z-angle is a metric derived from individual accelerometer components that is commonly used in sleep detection, and refers to the angle of the arm relative to the vertical axis of the body
        enmo - As calculated and described by the GGIR package, ENMO is the Euclidean Norm Minus One of all accelerometer signals, with negative values rounded to zero. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features
        test_series.parquet - Series to be used as the test data, containing the same fields as above. You will predict event occurrences for series in this file.
        train_events.csv - Sleep logs for series in the training set recording onset and wake events.
        series_id - Unique identifier for each series of accelerometer data in train_series.parquet.
        night - An enumeration of potential onset / wakeup event pairs. At most one pair of events can occur for each night.
        event - The type of event, whether onset or wakeup.
        step and timestamp - The recorded time of occurence of the event in the accelerometer series."""


role_prompt_suggestion = """
As a machine learning assistant, your task is to help users decide which machine learning approach is best suited for accomplishing their goal given some information about their data.
Simply return the top 5 types of machine learning approaches you would suggest without an explanation.
Format your response as "(<suggestion_1>, <suggestion_2>, <suggestion_3>, <suggestion_4>, <suggestion_5>)".
""".strip()

role_prompt_preprocess = f"""
As a python coding assistant, your task is to help users preprocess their data given some contextual information about the data and the suggested machine learning modeling approach.
Preprocessing will require you to analyze the column descriptions and values within the columns to build logic that prescribes datatypes among other data quality fixes.
Training series can be found at {train_test_path}. Training events can be found at ('cmi_sleep_states/train_events.csv').
Your response must be valid python code.
Format your response as:

```python
# code
```
""".strip()

role_prompt_features = """
As a machine learning assistant, your task is to help users build feature engineering python code to support their machine learning model selection and provided data information.
You will respond with valid python code that generates new features for the dataset that are appropriate for the "model_selection".
Reference passed preprocessing code when needed.
Address NaN or Null values that may arise in engineered features and address them as necessary. 
Address data types wherever possible, reference "A sample of the data".
Generate as many features as possible.
Format your response as:

```python
# code
```
All code should lie within a single: 
```python
# code
```
Do not return more than one:
```python
# code
``` 
""".strip()

role_prompt_model = f"""
As a machine learning assistant, your task is to help users write machine learning model code.
You will respond with valid python code that defines a machine learning solution.
Data information can be found in the context: data_summary. The model to write can be found in the context: model_selection. Preprocessing code can be found in the context: preprocessing_code_output. New features can be found in the context: raw_feature_output.
Training series can be found at {train_test_path}. Training events can be found at ('cmi_sleep_states/train_events.csv').
Use the preprocessing and feature engineering code provided.
Use XGBoost for decision trees, PyTorch for neural networks, and sklearn.
Always return an accuracy score.
Format your response as:

```python
# code
```
All code should lie within a single: 
```python
# code
```
Do not return more than one:
```python
# code
``` 
""".strip()

role_prompt_debug = """
As a python coding assistant, your task is to help users debug the supplied code using the context, code, and traceback provided.
Simply return the remedied code, but try to be proactive in debugging. If you see multiple errors that can be corrected, fix them all.
Format your response as:

```python
# code
```
""".strip()

def send_request_to_gpt(step_role, context, prompt, stream=False):

    # Handle string input for context
    if isinstance(context, str):
        context = [{"role": "user", "content": context}]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            # Establish the context of the conversation
            {
                "role": "system",
                "content": step_role,
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


def is_alpha_column(s):
    """
    Check if a Pandas Series only contains alphabetical strings or spaces.
    """
    return all(s.dropna().astype(str).apply(lambda x: x.replace(' ', '').isalpha()))


def preprocess_dataframe(df, unique_value_ratio=0.05):
    """
    Preprocess a Pandas DataFrame by inferring column data types and identifying anomalies.
    
    Parameters:
        df (Pandas DataFrame): The DataFrame to be preprocessed.
        unique_value_ratio (float): The unique-to-total values ratio to decide whether a column is categorical.
        
    Returns:
        DataFrame: A DataFrame with adjusted data types and filled NA/NaN values.
    """
    anomalies = {}
    
    # Detect anomalies based on Z-score for numerical columns
    for col in df.select_dtypes(include=[np.number]).columns:
        z_score = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = np.where(z_score > 3)[0]
        if len(outliers) > 0:
            anomalies[col] = len(outliers)

    # TODO: Add dynamic fillna support
    df.fillna(0.0, inplace=True)
    
    print(f"Anomaly Detection Summary: {anomalies}")
    
    return df


def dataframe_summary(df, dataset_description, sample_rows=5, sample_columns=14):
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

    # # Below code makes everything worse
    # # Adding a column data type summary
    # for col in df.columns:
        
    #     df[col].fillna('__MISSING__', inplace=True)
        
    #     try:
    #         df[col] = pd.to_datetime(df[col])
    #         continue
    #     except Exception:
    #         pass

    #     try:
    #         df[col] = pd.to_numeric(df[col], errors='raise')
    #         continue  
    #     except Exception:
    #         pass
        
    #     if is_alpha_column(df[col]):
    #         df[col] = df[col].astype('str')
    #     else:
    #         df[col] = df[col].astype('object')

    #     df[col].replace('__MISSING__', np.nan, inplace=True)
    
    # df.replace('__MISSING__', np.nan, inplace=True)
    # column_data_types = df.dtypes
    # column_data_types_str = "\n".join(f"- {col}: {dtype}" for col, dtype in column_data_types.items())

    # Basic summary statistics for numerical and categorical columns
    numerical_summary = df.describe(include=[np.number])
    
    has_categoricals = any(df.select_dtypes(include=['category', 'datetime', 'timedelta']).columns)

    if has_categoricals:
        categorical_summary = df.describe(include=['category', 'datetime', 'timedelta'])
    else:
        categorical_summary = pd.DataFrame(columns=df.columns)

    sampled = df.sample(min(sample_columns, df.shape[1]), axis=1).sample(min(sample_rows, df.shape[0]), axis=0)

    # Constructing a GPT-friendly output:
    output = (
        f"Here's a summary of the dataframe:\n"
        f"- Rows: {num_rows:,}\n"
        f"- Columns: {num_cols:,}\n\n"

        f"Column names and their descriptions:\n"
        f"{dataset_description}"

        f"Top columns with missing values:\n"
        f"{missing_values.to_string()}\n\n"

        f"Numerical summary:\n"
        f"{numerical_summary.to_string()}\n\n"

        f"A sample of the data ({sample_rows}x{sample_columns}):\n"
        f"{sampled.to_string()}"
    )
    
    return output

def extract_suggestions(response):
    """Extract suggestions from the model's response."""
    content = response["choices"][0]["message"]["content"]
    # Strip the surrounding parentheses and split by comma
    suggestions = content.strip('()').split(', ')
    return suggestions

def prompt_user_for_selection(suggestions):
    print("Please select one of the following suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(suggestions):
                return suggestions[choice - 1]
            else:
                print("Invalid choice. Please choose again.")
        except ValueError:
            print("Please enter a number.")

def update_context(existing_context, new_context, step):
    if step == 0:
        new_message = {"role": "user", "content": f"model_selection: {new_context}"}
    elif step == 0.5:
        new_message = {"role": "user", "content": f"preprocessing_code_output: {new_context}"}
    elif step == 1:
        new_message = {"role": "user", "content": f"raw_feature_output: {new_context}"}
    elif step == 2:
        new_message = {"role": "user", "content": f"raw_model_code_output: {new_context}"}
    else:
        raise ValueError("Invalid step provided")

    if isinstance(existing_context, str):
        existing_context = [{"role": "user", "content": f"data_summary: {existing_context}"}]

    existing_context.append(new_message)
    
    return existing_context

def print_context(context):
    for idx, message in enumerate(context):
        role = message['role']
        content = message['content']
        print(f"Message {idx + 1} ({role}):")
        print('-' * 30)  # separator for clarity
        print(content)
        print('\n' + '=' * 50 + '\n')  # separate different messages


def extract_code(message):
    substr = message.find('```python')
    incomplete_code = message[substr + 9 : len(message)]
    substr = incomplete_code.find('```')
    code = incomplete_code[0:substr]
    return code

def extract_content_from_gpt_response(response):
    return response['choices'][0]['message']['content']

def code_validation_agent(code):
    agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    # callback_manager=BaseCallbackManager,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps = True,
)  
    response = agent_executor({'input': f"Execute the code and make the necessary improvements. Do not stop iterating until the code runs without errors. Final output must be the entire code body with all changes: {code}"})
    print(json.dumps(response["output"], indent=2))

    return response["output"]

def choose_model(model_output_dict):
    """
    Prompt the user to select a model and return the corresponding output.
    
    Args:
    - model_output_dict (dict): Dictionary containing model names as keys and model output as values.
    
    Returns:
    - dict: Dictionary containing the selected model and its output.
    """
    print("Choose a Model:")
    for idx, model in enumerate(model_output_dict.keys(), 1):
        print(f"{idx}. {model}")
    
    choice = int(input("Enter the number corresponding to your model choice: "))
    
    # Validate choice
    while choice not in range(1, len(model_output_dict) + 1):
        print("Invalid choice. Please try again.")
        choice = int(input("Enter the number corresponding to your model choice: "))
    
    model_name = list(model_output_dict.keys())[choice - 1]
    return {model_name: model_output_dict[model_name]}


processed_train_series_set = preprocess_dataframe(train_series_set)   
df_context = dataframe_summary(processed_train_series_set, dataset_description)

suggestions = extract_suggestions(send_request_to_gpt(step_role=role_prompt_suggestion, context=df_context, prompt=goal_prompt))
model_output_dict = {}

for selected_content in suggestions:
    try:
        context_to_use = [{"role": "user", "content": f"data_summary: {df_context}"}]
        
        step_zero_context = update_context(context_to_use, selected_content, step=0)

        step_zero2_context = update_context(step_zero_context, extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_preprocess, context=step_zero_context, prompt="Generate preprocessing code for my dataset")), step=0.5)
        
        step_one_context = update_context(step_zero2_context, extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_features, context=step_zero2_context, prompt='Create new features for my dataset')), step=1)
        
        step_two_context = update_context(step_one_context, extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_model, context=step_one_context, prompt='Based on my model_selection, data_summary, and raw_feature_output. Output the machine learning model code')), step=2)
        
        print_context(step_two_context)
        
        output_code = extract_code(extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_model, context=step_one_context, prompt='Based on my model_selection, data_summary, and raw_feature_output. Output the machine learning model code')))

        if attempt_validation is True:
            code_validation_output = code_validation_agent(output_code)
        else:
            model_output_dict[selected_content] = code_validation_output

    except openai.error.InvalidRequestError as e:
        print(f"An error occurred: {e}. Skipping to next suggestion.")
        continue


print("Model Output Summary:")
for model, output in model_output_dict.items():
    print(f"Model: {model}")
    print(f"Output: {output}")
    print("-" * 50)

