import openai
import numpy as np
import pandas as pd
import tiktoken
import json
import os

# langchain imports
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI


# load testing data
recommender_system_set = pd.read_csv('C:/Users/willb/OneDrive/Documents/GitHub/placeholder1/test_data/recommender_set.csv')

# expose the key alot
openai.api_key = 'sk-'
os.environ["OPENAI_API_KEY"] = 'sk-'


role_prompt_zero = """
As a machine learning assistant, your task is to help users decide which machine learning approach is best suited for accomplishing their goal given some sample data.
Simply return the top 5 types of machine learning approaches you would suggest without an explanation.
Format your response as "(<suggestion_1>, <suggestion_2>, <suggestion_3>, <suggestion_4>, <suggestion_5>)"
""".strip()

role_prompt_one = """
As a machine learning assistant, your task is to help users build feature engineering python code to support their machine learning model selection and provided sample data.
You will respond with valid python code that generates new features for the dataset that are appropriate for the "model_selection".
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

role_prompt_two = """
As a machine learning assistant, your task is to help users write machine learning model code.
You will respond with valid python code that defines a machine learning solution.
Sample data can be found in the context: sample_data. The model to write can be found in the context: model_selection. New features can be found in the context: raw_feature_output.
Include the sample data in the code body.
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

def send_request_to_gpt(step_role, context, prompt, stream=False):

    # Handle string input for context
    if isinstance(context, str):
        context = [{"role": "user", "content": context}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
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



def dataframe_summary(df, sample_rows=5, sample_columns=5):
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

    column_info = pd.concat([df.dtypes, missing_values], axis=1)
    column_info.columns = ["Data Type", "Missing Values", "% Missing"]
    
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
        
        f"Top columns with missing values:\n"
        f"{missing_values.to_string()}\n\n"

        f"Numerical summary:\n"
        f"{numerical_summary.to_string()}\n\n"

        f"Categorical summary:\n"
        f"{categorical_summary.to_string()}\n\n"

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
    """Prompt the user to select a suggestion and return the selected value."""
    print("Please select one of the following suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    # Ensure user selection is valid
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
    # Determine the message based on the step
    if step == 0:
        new_message = {"role": "user", "content": f"model_selection: {new_context}"}
    elif step == 1:
        new_message = {"role": "user", "content": f"raw_feature_output: {new_context}"}
    elif step == 2:
        new_message = {"role": "user", "content": f"raw_model_code_output: {new_context}"}
    else:
        raise ValueError("Invalid step provided")

    # Check if existing_context is a string and convert to a list of dictionaries
    if isinstance(existing_context, str):
        existing_context = [{"role": "user", "content": f"sample_data: {existing_context}"}]

    # Append the new message to the existing context
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
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)  
    agent_executor.run(f"Test the following code to ensure it runs. If the code does not run iterate with improvements until it clears. Code to test: {code}")

df_context = dataframe_summary(recommender_system_set)

suggestions = extract_suggestions(send_request_to_gpt(step_role=role_prompt_zero, context=df_context, prompt='What kind of machine learning solution would you recommend for this dataset if I am looking to increase average checkout price'))
selected_content = prompt_user_for_selection(suggestions)

step_zero_context = update_context(df_context, selected_content, step=0)
step_one_context = update_context(step_zero_context, extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_one, context=step_zero_context, prompt='Create two new features for my dataset')), step=1)
step_two_context = update_context(step_one_context, extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_two, context=step_one_context, prompt='Based on my model_selection, sample_data, and raw_feature_output. Output the machine learning model code')), step=2)

output_code = extract_code(extract_content_from_gpt_response(send_request_to_gpt(step_role=role_prompt_two, context=step_one_context, prompt='Based on my model_selection, sample_data, and raw_feature_output. Output the machine learning model code')))
print_context(step_two_context)
print(output_code)
print(code_validation_agent(output_code))


