import io
import os
import sys
import marqo
import mlflow
import openai
import requests
import traceback
import subprocess
import webbrowser
import numpy as np
import pandas as pd
import mlflow.sklearn
from log_module import logger
from typing import Dict, List, Any
# from griptape.structures import Workflow
from contextlib import contextmanager
from docker_runtime import(
    check_for_image, 
    build_docker_image, 
    docker_runtime,
)
from context import(
    read_json_from_file,
    append_data_to_file,
    store_data_context,
    load_data_context
)
from tokens import (
    num_tokens_from_messages, 
    trim_messages_to_fit_token_limit,
)
from finetuning_set import dict_to_dataframe

# for locally hosted marqo client, vectorstore.py needs to be run and the container needs to be active
# log level output is commented out for notebook debugging (replace by print statements)

class Reach:
    def __init__(
            self,
            openai_api_key: str, 
            marqo_index: str, 
            train_set_path: str, 
            test_set_path: str, 
            dataset_description: str, 
            goal_prompt: str,
            attempt_validation: bool,
            **kwargs,
            ) -> None:
        self.openai_api_key = openai_api_key
        self.marqo_client = marqo.Client(url="http://localhost:8882")
        self.marqo_index = marqo_index
        self.context_file_name = "memory.txt"
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path
        self.dataset_description = dataset_description
        self.goal_prompt = goal_prompt
        self.attempt_validation = attempt_validation

        self.decision_preprompt = f"""
            As a decision making assistant, your task is to analyze the supplied user_goal and data_summary in the provided context to determine if the user_goal can be accomplished with a machine learning solution.
            Only return a single word response: 'yes' (all lowercase) if a machine learning solution is appropriate.
            Otherwise return an explanation as to why a machine learning solution is not a good approach.
            """

        self.data_analyst_preprompt = f"""
            As a data analyst and python coding assistant, your task is to develop python code to help users answer their question or accomplish their goal.
            Generate the necessary python code to answer the supplied prompt.
            Data can be found at {self.train_set_path}.

            ```python
            # code
            ```
            """.strip()
        
        self.preprocess_for_llm_preprompt = f"""
            As a python coding assistant, your task is to use statistics and simple descriptive outputs to generate some understanding of a dataset given a light description.
            Leverage common exploratory data analysis and statistics packages such as numpy, pandas, etc, to output information about the dataset.              
            Data can be found at {self.train_set_path}.
            Your response must be valid python code.
            Format your response as:

            ```python
            # code
            ```
            """.strip()
        self.feature_engineering_preprompt = """
            As a machine learning assistant, your task is to help users build feature engineering python code to support their machine learning model selection and provided data information.
            Utilize the preprocessing_context in context for information about the dataset.
            You will respond with valid python code that generates new features for the dataset that are appropriate for the model_selection and the user_goal provided in the context.
            Generate as many features as possible, generate AT LEAST 3 new features.
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
        self.model_development_preprompt = f"""
            As a machine learning assistant, your task is to help users write machine learning model code.
            You will respond with valid python code that defines a machine learning solution.
            Data information can be found in the context: data_summary, and preprocessing_context. The goal of the model can be found in: user_goal. And necessary feature engineering in: feature_engineering_code.
            Training data can be found at {self.train_set_path}.
            Use the feature engineering code provided.
            Use XGBoost for decision trees, PyTorch for neural networks, and sklearn.
            Always return an accuracy score and a model results dataframe with descriptive columns.
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
        self.performance_eval_preprompt = f"""
            As a python coding assistant, your task is to help users add additional outputs to their machine learning model code to improve their understanding of the performed analysis.
            Update the supplied machine learning model code with model performance evaluation logic such as, but not limited to, feature importance, ROC curves, etc.
            Prioritze generating new outputs as dataframes or strings and not visualizations.
            Format your response as:

            ```python
            # code
            ```
            """.strip()
        self.validation_preprompt = f"""
            As a python coding assistant, your task is to help users debug the supplied code using the context, code, and traceback provided.
            Simply return the remedied code, but try to be proactive in debugging. If you see multiple errors that can be corrected, fix them all.
            Training data can be found at {self.train_set_path}.
            You must return THE ENTIRE ORIGINAL CODE BLOCK WITH THE REQUIRED CHANGES.
            Format your response as:

            ```python
            # code
            ```
            """.strip()
        #TODO improve
        self.so_what_description_preprompt = f"""
            As a data analysis assistant, your task is to interpret the users goal, and outputs generated by supplied code to generate relevant insights.
            Reference the supplied user_goal, data_summary, and analysis_code in the context.
            Ideally these insights are actionable, that is to say, the user can leverage generated outputs to answer their question or accomplish their goal.
            """.strip()
        self.openai_api_key = openai_api_key

        self.log = logger()

    def add_index(
            self,
            index_name: str, 
            model: str = 'sentence-transformers/all-MiniLM-L6-v2'
            ) -> None:
        
        self.marqo_client.create_index(index_name, model)

    def store_text(
            self,
            index_name: str, 
            text_to_store_title: str, 
            text_to_store: str
            ) -> None:
        
        self.marqo_client.index(index_name).add_documents(
            [
                {
                    "Title": text_to_store_title,
                    "Description": text_to_store
                }
            ],
            tensor_fields=["Title", "Description"]
        )

    def query_marqo_db(self, index_name: str, search_query: str) -> Dict[str, Any]:
        query = self.marqo_client.index(index_name).search(q=search_query, searchable_attributes=["Title", "Description"])

        return query
    
    def send_request_to_gpt(
            self,
            role_preprompt: str, 
            prompt: str,
            context: Dict[str, str],  
            stream: bool = False
            ) -> (Any | List | Dict):

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
    
    def preprocess_dataframe(self, unique_value_ratio: float = 0.05) -> pd.DataFrame:
        """
        Preprocess a Pandas DataFrame by inferring column data types and identifying anomalies.
        
        Parameters:
            df (Pandas DataFrame): The DataFrame to be preprocessed.
            unique_value_ratio (float): The unique-to-total values ratio to decide whether a column is categorical.
            
        Returns:
            DataFrame: A DataFrame with adjusted data types and filled NA/NaN values.
        """
        anomalies = {}

        if ".parquet" in self.train_set_path:
            df = pd.read_parquet(self.train_set_path)
        elif ".csv" in self.train_set_path:
            df = pd.read_csv(self.train_set_path)
        
        # Detect anomalies based on Z-score for numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            z_score = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = np.where(z_score > 3)[0]
            if len(outliers) > 0:
                anomalies[col] = len(outliers)

        # TODO: Add dynamic fillna support
        df.fillna(0.0, inplace=True)
        
        print(f"Anomaly Detection Summary: {anomalies}")
        # self.log.info(f"Anomaly Detection Summary: {anomalies}")
        
        return df


    def dataframe_summary(
            self, 
            df: pd.DataFrame, 
            dataset_description: str = None, 
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
    
    def generate_suggestion_text(self, n_suggestions: int) -> str:

        return ', '.join(['(<suggestion_{0}>)'.format(i) for i in range(1, n_suggestions+1)])

    def extract_suggestions(
            self,
            response: (Any | List | Dict),
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
            response: (Any | List | Dict)
            ) -> str:
        return response['choices'][0]['message']['content']
    
    
    def code_validation_agent(
            self, 
            code_to_validate: str, 
            context: List[Dict[str, str]], 
            max_attempts: int = 10,
        ) -> str:

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

                # Debugging via GPT-4
                response = self.send_request_to_gpt(
                    role_preprompt=self.validation_preprompt,
                    context=context,
                    prompt=f"""
                    Debug the following python code: {code_to_validate}. \n\nError:\n{error_message}\n\nTraceback:\n{error_traceback}\n\n.
                    Training data can be found at {self.train_set_path} 
                    You must return THE ENTIRE ORIGINAL CODE BLOCK WITH THE REQUIRED CHANGES.
                    """,
                )

                suggestion = self.extract_code(
                    (self.extract_content_from_gpt_response(
                        response
                        )
                    )
                )

                print(f"Updated Code: \n{suggestion}")
                # self.log.info(f"Updated Code: \n{suggestion}")

                code_to_validate = suggestion
                attempts += 1

        print("Max attempts reached. Code is still broken.")
        # self.log.info("Max attempts reached. Code is still broken.")
        return None


    def mlflow_integration(self, validated_model_code: str, model_name: str) -> None:
        
        mlflow.autolog()

        with mlflow.start_run() as run:

            mlflow.set_tag("model_name", model_name)
            try:
                exec(validated_model_code)
            except Exception as e:
                error_message = str(e)
                print(f"Upstream failure with returned model code: {error_message}")
                pass

            model_uri = f"runs:/{run.info.run_uuid}/model"
            registered_model_name = model_name
            mlflow.register_model(model_uri, registered_model_name)


    def launch_mlflow_ui(self, port: int = 5000) -> subprocess.Popen[bytes]:
        """Launch MLflow UI in a separate process."""
        cmd = ["mlflow", "ui", "--port", str(port)]
        process = subprocess.Popen(cmd)
        webbrowser.open(f'http://127.0.0.1:{port}', new=2, autoraise=True)

        return process
    

    def serve_mlflow_model(self, run_id: str, port: int = 5000) -> subprocess.Popen[bytes]:
        """Serve an MLflow model in a separate process."""
        model_uri = f"runs:/{run_id}/model"
        cmd = ["mlflow", "models", "serve", "-m", model_uri, "--port", str(port)]
        process = subprocess.Popen(cmd)
        return process
    
    
    def submit_mlflow_prediction(self, input_data: pd.DataFrame, port: int = 5000) -> dict:
        url = f'http://127.0.0.1:{port}/invocations'
        headers = {
            "Content-Type": "application/json; format=pandas-split"
        }

        data = input_data.to_json(orient="split")

        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code != 200:
            raise ValueError("Prediction request failed with status: {}".format(response.status_code))
            
        return response.json()

    @contextmanager
    def capture_stdout(self) -> io.StringIO:
        original_stdout = sys.stdout
        buffer = io.StringIO()
        sys.stdout = buffer
        try:
            yield buffer
        finally:
            sys.stdout = original_stdout

    def so_what_description(self, context: List[Dict[str, str]],  validated_model_code: str) -> str:
        with self.capture_stdout() as buffer:
            try:
                exec(validated_model_code)
            except Exception as e:
                error_message = str(e)
                print(f"Upstream failure with returned model code: {error_message}")
                pass

        captured_out = buffer.getvalue()

        insight = self.extract_content_from_gpt_response(
            self.send_request_to_gpt(
                role_preprompt=self.so_what_description_preprompt,
                prompt=f"Given the output of the following code: {validated_model_code}. Output: {captured_out}. What insights can you extract from the model's analysis",
                context=context,
            )
        )

        return insight
        

    def main(self, n_suggestions: int, index_name: str) -> None:
        # workflow = Workflow()

        if os.path.exists("data_context.txt"):

            # self.log.info("Loading data context")
            print("Loading data context")
            preprocessing_context, df_context, validated_preprocessing_code = load_data_context()
        
        else:

            # self.log.info("Interpreting the provided data")
            print("Interpreting the provided data")
            processed_train_data = self.preprocess_dataframe()  

            df_context = self.dataframe_summary(
                processed_train_data, 
                self.dataset_description
            )
            preprocessing_code = self.extract_code(
                        self.extract_content_from_gpt_response(
                            self.send_request_to_gpt(
                                role_preprompt=self.preprocess_for_llm_preprompt, 
                                context=[{"role": "user", "content": f"data_summary: {df_context}"}], 
                                prompt="Help me understand the information contained in the dataset."
                            )
                        )
                    )

            validated_preprocessing_code = self.code_validation_agent(
                preprocessing_code,
                context=[
                    {"role": "user", "content": f"data_summary: {df_context}"},
                ],
                max_attempts=10,
            )

            #This is a hacky solution
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            #TODO marking exec statement
            exec(validated_preprocessing_code)
            preprocessing_context = buffer.getvalue()
            sys.stdout = old_stdout

            # self.log.info("Storing data context")
            print("Storing data context")
            store_data_context(preprocessing_context, df_context, validated_preprocessing_code)

        memory_dict = read_json_from_file('memory.txt')

        decision = self.extract_content_from_gpt_response(
                    self.send_request_to_gpt(
                        role_preprompt=self.decision_preprompt, 
                        context=[
                            {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                            {"role": "user", "content": f"data_summary: {df_context}"},
                            {"role": "user", "content": f"memory: {memory_dict}"},
                            ], 
                        prompt="Analyze the supplied user_goal, data_summary, and memory in the context and decide if machine learning is necessary to answer the question. Only return a single word respone, yes or no. Always check the memory for relevant information."
                    )
                )
        
        print(f'Decision: {decision}')

        if decision == 'yes':
            # self.log.info('ML modelling is required. Beginning model development')
            print('ML modelling is required. Beginning model development')

            # # There's probably a smarter way to do this
            # token_count_ml = num_tokens_from_messages(
            #         (
            #             {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
            #             {"role": "user", "content": f"data_summary: {df_context}"},
            #             {"role": "user", "content": f"feature_engineering_code: {self.extract_code(feature_engineering_context)}"},
            #             {"role": "user", "content": f"memory: {memory_dict}"}
            #         )
            # )

            # if token_count_ml > 8192:
            #     trimmed_message = trim_messages_to_fit_token_limit(
            #         (
            #             {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
            #             {"role": "user", "content": f"data_summary: {df_context}"},
            #             {"role": "user", "content": f"feature_engineering_code: {self.extract_code(feature_engineering_context)}"},
            #             {"role": "user", "content": f"memory: {memory_dict}"}
            #         )
            # )
            #     self.log.info(f'Trimmed message: {trimmed_message}')
            #     print(trimmed_message)

            suggestion_text = self.generate_suggestion_text(n_suggestions)
            self.suggestion_preprompt = f"""
                As a machine learning assistant, your task is to help users decide which machine learning approach is best suited for accomplishing their goal given some information about their data.
                Simply return the top {n_suggestions} types of machine learning approaches you would suggest without an explanation.
                Format your response as "{suggestion_text}".
                """.strip()

            suggestions = self.extract_suggestions(
                    self.send_request_to_gpt(
                        role_preprompt=self.suggestion_preprompt, 
                        context=df_context, 
                        prompt=self.goal_prompt
                    )
                )

            for model in suggestions:

                feature_engineering_context = self.extract_content_from_gpt_response(
                        self.send_request_to_gpt(
                            role_preprompt=self.feature_engineering_preprompt, 
                            context=[
                                {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                                {"role": "user", "content": f"model_selection: {model}"},
                                {"role": "user", "content": f"data_summary: {df_context}"},
                                {"role": "user", "content": f"preprocessing_context: {preprocessing_context}"},

                            ], 
                            prompt="Create new features for my dataset based on my user_goal, model_selection, and data_summary available in context."
                        )
                    )
                print(f"feature engineering context: {feature_engineering_context}")

                model_context = self.extract_content_from_gpt_response(
                        self.send_request_to_gpt(
                            role_preprompt=self.feature_engineering_preprompt, 
                            context=[
                                {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                                {"role": "user", "content": f"data_summary: {df_context}"},
                                {"role": "user", "content": f"preprocessing_context: {preprocessing_context}"},
                                {"role": "user", "content": f"feature_engineering_code: {feature_engineering_context}"},
                                {"role": "user", "content": f"memory: {memory_dict}"}
                            ], 
                            prompt="Based on my user_goal, data_summary, preprocessing_context, and feature_engineering_code. Generate the machine learning model code, be sure to utilize the feature engineering code provided in the context."
                        )
                    )
                model_context_performance_metric_additions = self.extract_content_from_gpt_response(
                        self.send_request_to_gpt(
                            role_preprompt=self.performance_eval_preprompt, 
                            context=[
                                {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                                {"role": "user", "content": f"data_summary: {df_context}"},
                            ], 
                            prompt=f"""Based on my data_summary, user_goal, and the following code: {self.extract_code(model_context)}, 
                            update the code to include model performance evaluations to help me understand the insights the model is generating.
                            Training data can be found at {self.train_set_path}.
                            You must return THE ENTIRE ORIGINAL CODE BLOCK WITH THE ADDITIONS.
                            """
                        )
                    )
                
                if self.attempt_validation == True:
                    validated_code = self.code_validation_agent(
                        code_to_validate=self.extract_code(model_context_performance_metric_additions),
                        context=[
                            {"role": "user", "content": f"data_summary: {df_context}"},
                        ],
                        max_attempts=10,
                    )

                # self.add_index(index_name=index_name)

                # self.store_text(
                #     index_name=index_name, 
                #     text_to_store_title=f"{model}_data_summary", 
                #     text_to_store=df_context
                # )

                # self.store_text(
                #     index_name=index_name, 
                #     text_to_store_title=f"{model}_preprocessing", 
                #     text_to_store=preprocess_context
                # )

                # self.store_text(
                #     index_name=index_name,
                #     text_to_store_title=f"{model}_preprocessing_code",
                #     text_to_store=self.extract_code(preprocess_context)
                # )

                # self.store_text(
                #     index_name=index_name, 
                #     text_to_store_title=f"{model}_feature_engineering", 
                #     text_to_store=feature_engineering_context
                # )

                # self.store_text(
                #     index_name=index_name,
                #     text_to_store_title=f"{model}_feature_engineering_code",
                #     text_to_store=self.extract_code(feature_engineering_context)
                # )
                
                # if self.attempt_validation == True:
                #     self.store_text(
                #         index_name=index_name,
                #         text_to_store_title=f"{model}_model_code",
                #         text_to_store=validated_code
                #     )
                # else:
                #     self.store_text(
                #         index_name=index_name, 
                #         text_to_store_title=f"{model}_model", 
                #         text_to_store=model_context
                #     )

                #     self.store_text(
                #         index_name=index_name,
                #         text_to_store_title=f"{model}_model_code",
                #         text_to_store=self.extract_code(model_context)
                #     )
                
                print(f"Validated model code for {model}: {validated_code}")
                # self.log.info(f"Validated model code for {model}: {validated_code}")

                #TODO weird things happening with MLFlow bogging runs, likely a local caching issue
                #that will require some level of artifact cleaning or garbage collection.
                self.mlflow_integration(
                    model_name=model,
                    validated_model_code=validated_code,
                )
                
                if self.attempt_validation == True:
                    so_what = self.so_what_description(
                        context=[
                            {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                            {"role": "user", "content": f"analysis_code: {validated_code}"},
                        ],
                        validated_model_code=validated_code
                    )
                
                #TODO so_what return type is str | unbound, need to investigate this
                print(so_what)
                # self.log.info(so_what)

            #TODO weird things happening with MLFlow bogging runs, likely a local caching issue
            #that will require some level of artifact cleaning or garbage collection.    
            self.launch_mlflow_ui(port = 5000)

            dict_to_dataframe(
                data_dict = {
                    'goal': self.goal_prompt,
                    'data_summary': df_context,
                    'preprocessing_code': validated_preprocessing_code,
                    'feature_engineering_code': feature_engineering_context,
                    'model_code': validated_code,
                    'analysis_code': None,
                    'ml_model': 1
                    },
                file_path = 'finetuning_set.csv'
                )
        else:
            # self.log.info('No modelling is required. Beginning analysis')

            # # There's probably a smarter way to do this
            # token_count_ml = num_tokens_from_messages(
            #         (
            #             {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
            #             {"role": "user", "content": f"data_summary: {df_context}"},
            #             {"role": "user", "content": f"memory: {memory_dict}"}
            #         )
            # )

            # if token_count_ml > 8192:
            #     trimmed_message = trim_messages_to_fit_token_limit(
            #         (
            #             {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
            #             {"role": "user", "content": f"data_summary: {df_context}"},
            #             {"role": "user", "content": f"memory: {memory_dict}"}
            #         )
            # )
            #     self.log.info(f'Trimmed message: {trimmed_message}')
            #     print(trimmed_message)

            print('No modelling is required. Beginning analysis')
        
            analysis_response_context = self.extract_content_from_gpt_response(
                    self.send_request_to_gpt(
                        role_preprompt=self.data_analyst_preprompt, 
                        context=[
                            {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                            {"role": "user", "content": f"data_summary: {df_context}"},
                            {"role": "user", "content": f"preprocessing_context: {preprocessing_context}"},
                            {"role": "user", "content": f"memory: {memory_dict}"}
                            ], 
                        prompt="Analyze the supplied user_goal, data_summary, preprocessing_context, and memory in the context and generate python code that, when run, answers my question provided in user_goal in the context. Be sure to always check in the memory section of the context for relevant information."
                    )
                )
            
            if self.attempt_validation == True:
                    validated_code = self.code_validation_agent(
                        code_to_validate=self.extract_code(analysis_response_context),
                        context=[
                            {"role": "user", "content": f"data_summary: {df_context}"},
                        ],
                        max_attempts=10,
                    )

            if self.attempt_validation == True:
                    so_what = self.so_what_description(
                        context=[
                            {"role": "user", "content": f"user_goal: {self.goal_prompt}"},
                            {"role": "user", "content": f"data_summary: {df_context}"},
                            {"role": "user", "content": f"analysis_code: {validated_code}"},
                        ],
                        validated_model_code=validated_code
                    )

                
            #TODO so_what return type is str | unbound, need to investigate this
            print(so_what)

        #TODO cleaning: below can be packed into a store_code_output function
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        #TODO marking exec statement
        # if the only code output is an image, nothing will be added to output
        exec(validated_code)
        print(validated_code)
        code_output = buffer.getvalue()
        sys.stdout = old_stdout

        dict_to_dataframe(
                data_dict = {
                    'goal': self.goal_prompt,
                    'data_summary': df_context,
                    'preprocessing_code': validated_preprocessing_code,
                    'feature_engineering_code': None,
                    'model_code': None,
                    'analysis_code': validated_code,
                    'ml_model': 0,
                    },
                file_path = 'finetuning_set.csv'
        )

        if os.path.exists("memory.txt") or not os.path.exists("memory.txt"):
            append_data_to_file(
                filename='memory.txt',
                data={
                    "user_goal": self.goal_prompt,
                    "solution": validated_code,
                    "output": code_output,
                }
            )       
