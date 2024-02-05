import streamlit as st
import openai
import os
import sys
import argparse
sys.path.append('./lats')
from lats_main import lats_main

st.set_page_config(layout="wide")

# Initialize session state variables if they don't exist.
if 'response_content' not in st.session_state:
    st.session_state.response_content = None

# Creating main columns for the chat and runtime notifications
chat_col = st.container()

sidebar = st.sidebar
# Runtime Section
runtime_container = st.container()

# Parameters Section
sidebar.title("**A lapis.rocks Project** (https://lapis.rocks/)")
parameters_section = sidebar.expander("Parameters", expanded=False)
tree_width = parameters_section.number_input("Tree Width", min_value=1, max_value=5, value=1)
tree_depth = parameters_section.number_input("Tree Depth", min_value=1, max_value=8, value=3)
iterations = parameters_section.number_input("Iterations", min_value=1, max_value=4, value=2)
key = st.sidebar.text_input("Enter your OpenAI Api Key:", type="password")
sidebar.markdown('<hr style="margin-top: 0.5rem; margin-bottom: 0.5rem;">', unsafe_allow_html=True)

with sidebar:
    runtime_container = st.container()
    runtime_container.empty()

runtime_messages = []

def make_args(instruction, tree_depth, tree_width, iterations):
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

def run_querry():
    if user_input:
        
        # Create a new container for each subsequent message
        runtime_container.write("Initiating process...")

        # Make it so that prints go to runtime_container writes instead
        old_stdout = sys.stdout
        sys.stdout = runtime_container

        with chat_col:

            with st.spinner('Running...'):
                args = make_args(user_input, tree_depth, tree_width, iterations)
                # main call
                response = lats_main(args)

        sys.stdout = old_stdout
        runtime_container.write("Response fetched.")
        chat_col.markdown('<hr style="margin-top: 0.5rem; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
        chat_col.write(f"```python\n{response} \n")

        return response

# User input section at the bottom of the page
with chat_col:
    user_input = st.text_area("Enter your message here:", placeholder="Type your message here...", label_visibility="collapsed")
    button = st.button("Send")

    if button:
        fail = False
        if key == "":
            st.warning("Missing OpenAI API Key")
            fail = True

        if user_input == "":
            st.warning("Missing a coding problem")
            fail = True
        
        if (not fail):
            openai.api_key = key
            run_querry()