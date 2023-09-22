import os
import marqo
import openai
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from griptape.structures import Workflow

test_set = pd.read_parquet('cmi_sleep_states/test_series.parquet')
test_set_path = 'cmi_sleep_states/test_series.parquet'
train_set = pd.read_parquet('C:/Users/willb/Downloads/train_series.parquet')
train_set_path = 'C:/Users/willb/Downloads/train_series.parquet'
train_events_set = pd.read_csv('cmi_sleep_states/train_events.csv')
attempt_validation = False

"""
Current State:
main will run a sequential pipeline similar to the prototype solution where
each suggestion is iterated through, preprocessing, feature engineering, and model code
will be produced. Everything is then stored in marqo, and can be queried.

Next Steps:
Parallelize tasks in main: each suggestion pipeline should run at the same time
so running 5 different generation builds takes the same time as 1. This will
likely take some trial and error to figure out how many tasks can be run without
exceeding the rate limit.
"""

# expose the key alot
openai.api_key = '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'
os.environ["OPENAI_API_KEY"] = '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'

goal_prompt = "What kind of machine learning solution would you recommend for this dataset if I am looking to detect sleep onset and wake. You will develop a model trained on wrist-worn accelerometer data in order to determine a persoms sleep state."

dataset_description = """
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
suggestion_preprompt = """
As a machine learning assistant, your task is to help users decide which machine learning approach is best suited for accomplishing their goal given some information about their data.
Simply return the top 5 types of machine learning approaches you would suggest without an explanation.
Format your response as "(<suggestion_1>, <suggestion_2>, <suggestion_3>, <suggestion_4>, <suggestion_5>)".
""".strip()

# need to think of a way to f string in paths to multiple data sets if passed in kwargs.
preprocess_preprompt = f"""
As a python coding assistant, your task is to help users preprocess their data given some contextual information about the data and the suggested machine learning modeling approach.
Preprocessing will require you to analyze the column descriptions and values within the columns to build logic that prescribes datatypes among other data quality fixes.
Training series can be found at {train_set_path}.
Your response must be valid python code.
Format your response as:

```python
# code
```
""".strip()

feature_engineering_preprompt = """
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

model_development_preprompt = f"""
As a machine learning assistant, your task is to help users write machine learning model code.
You will respond with valid python code that defines a machine learning solution.
Data information can be found in the context: data_summary. The model to write can be found in the context: model_selection. Preprocessing code can be found in the context: preprocessing_code_output. New features can be found in the context: raw_feature_output.
Training series can be found at {train_set_path}.
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

validation_preprompt = """
As a python coding assistant, your task is to help users debug the supplied code using the context, code, and traceback provided.
Simply return the remedied code, but try to be proactive in debugging. If you see multiple errors that can be corrected, fix them all.
Format your response as:

```python
# code
```"""


# for locally hosted marqo client, vectorstore.py needs to be run and the container needs to be active
mq = marqo.Client(url="http://localhost:8882")

class Reach:
    def __init__(
            self, 
            marqo_client: marqo.Client,
            marqo_index: str, 
            train_set: pd.DataFrame, 
            test_set: pd.DataFrame, 
            dataset_description: str, 
            goal_prompt: str,
            attempt_validation: bool,
            suggestion_preprompt: str,
            preprocess_preprompt: str,
            feature_engineering_preprompt: str,
            model_development_preprompt: str,
            validation_preprompt: str,
            **kwargs,
            ) -> None:
        self.marqo_client = marqo_client
        self.marqo_index = marqo_index
        self.train_set = train_set
        self.test_set = test_set
        self.dataset_description = dataset_description
        self.goal_prompt = goal_prompt
        self.attempt_validation = attempt_validation
        self.suggestion_preprompt = suggestion_preprompt
        self.preprocess_preprompt = preprocess_preprompt
        self.feature_engineering_preprompt = feature_engineering_preprompt
        self.model_development_preprompt = model_development_preprompt
        self.validation_preprompt = validation_preprompt


    def add_index(
            self,
            index_name: str, 
            model: str = 'sentence-transformers/all-MiniLM-L6-v2'
            ) -> None:
        
        mq.create_index(index_name, model)

    def store_text(
            self,
            index_name: str, 
            text_to_store_title: str, 
            text_to_store: str
            ) -> None:
        
        mq.index(index_name).add_documents(
            [
                {
                    "Title": text_to_store_title,
                    "Description": text_to_store
                }
            ],
            tensor_fields=["Title", "Description"]
        )

    def query_marqo_db(self, index_name: str, search_query: str) -> Dict[str, Any]:
        query = mq.index(index_name).search(q=search_query)

        return query
    
    def send_request_to_gpt(
            self,
            role_preprompt: str, 
            context: Dict[str, str], 
            prompt: str, 
            stream: bool = False
            ) -> openai.Generator[Any | list | dict, None, None] | Any | list | dict:

        # Handle string input for context
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]

        response = openai.ChatCompletion.create(
            model="gpt-4",
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
    
    def preprocess_dataframe(self, df: pd.DataFrame, unique_value_ratio: float = 0.05) -> pd.DataFrame:
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


    def dataframe_summary(
            self, 
            df: pd.DataFrame, 
            dataset_description: str, 
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
    
    def extract_suggestions(
            self,
            response: openai.Generator[Any | list | dict, None, None] | Any | list | dict
            ) -> List[str]:
        
        content = response["choices"][0]["message"]["content"]
        suggestions = content.strip('()').split(', ')

        return suggestions
    
    def extract_code(self, message: str) -> str:
        substr = message.find('```python')
        incomplete_code = message[substr + 9 : len(message)]
        substr = incomplete_code.find('```')
        code = incomplete_code[0:substr]
        return code

    def extract_content_from_gpt_response(
            self,
            response: openai.Generator[Any | list | dict, None, None] | Any | list | dict
            ) -> str:
        return response['choices'][0]['message']['content']
    
    def main(self, index_name: str) -> None:
        workflow = Workflow()

        processed_train_data = self.preprocess_dataframe(self.train_set)  

        df_context = self.dataframe_summary(
            processed_train_data, 
            self.dataset_description
        )

        suggestions = self.extract_suggestions(
                self.send_request_to_gpt(
                    role_preprompt=self.suggestion_preprompt, 
                    context=df_context, 
                    prompt=goal_prompt
                )
            )

        for model in suggestions:
            preprocess_context = self.extract_content_from_gpt_response(
                    self.send_request_to_gpt(
                        step_role=preprocess_preprompt, 
                        context=[{"role": "user", "content": f"data_summary: {df_context}"}], 
                        prompt="Generate preprocessing code for my dataset"
                    )
                )
    
            feature_engineering_context = self.extract_content_from_gpt_response(
                    self.send_request_to_gpt(
                        step_role=feature_engineering_preprompt, 
                        context=[
                            {"role": "user", "content": f"data_summary: {df_context}"},
                            {"role": "user", "content": f"preprocess_context: {preprocess_context}"},
                        ], 
                        prompt="Create new features for my dataset"
                    )
                )
            model_context = self.extract_content_from_gpt_response(
                    self.send_request_to_gpt(
                        step_role=feature_engineering_preprompt, 
                        context=[
                            {"role": "user", "content": f"data_summary: {df_context}"},
                            {"role": "user", "content": f"preprocess_context: {preprocess_context}"},
                            {"role": "user", "content": f"feature_engineering_context: {feature_engineering_context}"},
                        ], 
                        prompt="Based on my model_selection, data_summary, and raw_feature_output. Output the machine learning model code"
                    )
                )

            self.add_index(index_name=index_name)

            self.store_text(
                index_name=index_name, 
                text_to_store_title=f"{model}_data_summary", 
                text_to_store=df_context
            )

            self.store_text(
                index_name=index_name, 
                text_to_store_title=f"{model}_preprocessing", 
                text_to_store=preprocess_context
            )

            self.store_text(
                index_name=index_name,
                text_to_store_title=f"{model}_preprocessing_code",
                text_to_store=self.extract_code(preprocess_context)
            )

            self.store_text(
                index_name=index_name, 
                text_to_store_title=f"{model}_feature_engineering", 
                text_to_store=feature_engineering_context
            )

            self.store_text(
                index_name=index_name,
                text_to_store_title=f"{model}_feature_engineering_code",
                text_to_store=self.extract_code(feature_engineering_context)
            )
            
            self.store_text(
                index_name=index_name, 
                text_to_store_title=f"{model}_model", 
                text_to_store=model_context
            )

            self.store_text(
                index_name=index_name,
                text_to_store_title=f"{model}_model_code",
                text_to_store=self.extract_code(model_context)
            )
            
            print(f"Initial model code for model {model}: {self.extract_code(model_context)}")