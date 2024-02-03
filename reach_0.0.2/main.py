# this is more of a testing ground right now
import sys
import openai
import argparse
import pandas as pd
sys.path.append('./agent_core/tree_search')
from lats_main import lats_main
from utils import dataframe_summary
from prompts import *

openai.api_key = ""

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
analyst_preprompt = data_analyst_preprompt()
feature_engineering_prompt = feature_engineering_preprompt()
model_development_prompt = model_development_preprompt()

packages = available_packages_prompt()
ml = False

if not ml:
    args = build_args(instruction=f'Your role: {analyst_preprompt}, {packages}. User prompt: {user_prompt}', tree_depth=3, tree_width=2, iterations=2)
    response = lats_main(args)

    print(f'final response: {response}')
else:
    feature_engineering_args = build_args(instruction=f'Your role: {feature_engineering_prompt}, {packages}. User prompt: {user_prompt}', tree_depth=3, tree_width=2, iterations=2)
    feature_engineering_response = lats_main(feature_engineering_args)
    print(f'f_eng response: {feature_engineering_response}')
    model_development_args = build_args(instruction=f'Your role: {model_development_prompt}, {packages}, Available feature_engineering: {feature_engineering_response}. User prompt: {user_prompt}', tree_depth=3, tree_width=2, iterations=2)
    model_development_resposne = lats_main(model_development_args)

    print(f'final response: {model_development_resposne}')