# this is more of a testing ground right now
import sys
import openai
import argparse
import pandas as pd
sys.path.append('./agent_core/tree_search')
from lats_main import lats_main
from utils import dataframe_summary
from prompts import *

openai.api_key = "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"

def build_args(instruction, tree_depth, tree_width, iterations):
    parser = argparse.ArgumentParser()

    parser.add_argument("--strategy", default="mcts", help="Strategy to use")
    parser.add_argument("--language", default="py", help="Programming language")
    parser.add_argument("--model", default="gpt-4", help="Model type")
    parser.add_argument("--max_iters", default=iterations, help="Maximum iterations")
    parser.add_argument("--instruction", default=instruction, help="Instruction text")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")  # Temporary
    parser.add_argument("--n_samples", type=int,
                        help="The number of nodes added during expansion", default=tree_width)
    parser.add_argument("--depth", type=int,
                        help="Tree depth", default=tree_depth)
    args = parser.parse_args()
    return args

test_df = pd.read_csv('C:/Users/willb/OneDrive/Documents/GitHub/placeholder1/synthetic_sets/graphics_card_spec.csv')
text_repr_dataframe = dataframe_summary(
    df = test_df,
    dataset_description=None,
    sample_rows=10,
    sample_columns=len(test_df.columns),
)


user_prompt = 'return statistical analysis of the dataset located at: C:/Users/willb/OneDrive/Documents/GitHub/placeholder1/synthetic_sets/graphics_card_spec.csv'
preprompt = data_analyst_preprompt()
packages = available_packages_prompt()

args = build_args(instruction=f'Your role: {preprompt}, {packages}. User prompt: {user_prompt}', tree_depth=3, tree_width=2, iterations=2)
response = lats_main(args)

print(f'final response: {response}')