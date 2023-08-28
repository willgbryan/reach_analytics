from token_limit import trim_messages_to_fit_token_limit
from context import summarize_dataframe, summarize_series
from generate import generate_next_from_history
from preprompt_store import STEP_ZERO_PROMPT
import pandas as pd
import openai

openai.api_key = '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'

base_prompt = "I would like to improve average checkout price per cart."

dummy_df = pd.read_csv('C:/Users/willb/OneDrive/Documents/GitHub/placeholder1/test_data/recommender_set.csv')

context_df = summarize_dataframe(dummy_df, sample_rows=10, sample_columns=6)
prompt = STEP_ZERO_PROMPT + base_prompt

# context needs to be type: List[Dict[str, str]]
"""
Idea will be to build this context step by step starting
with the opening prompt and some sample data.

Step 0:
context = [{'data' : context_df}]

Step 1:
context = [{'data' : context_df}, {'model': 'recommender system'}]

Step 2:
context = [{'data' : context_df}, {'model': 'recommender system'}, {'features': <code>}]

and so on....

"""
context = [{'data' : context_df}]
generate_next_from_history(context, prompt, stream=False)

