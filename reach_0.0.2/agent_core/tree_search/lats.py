# from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
import sys
sys.path.append('C:/Users/willb/OneDrive/Documents/GitHub/placeholder1/reach_0.0.2/agent_core')
from executors import executor_factory
from generators import generator_factory, model_factory
from typing import List, Dict, Any
import math
from typing import Tuple
import sys
import random

sys.set_int_max_str_digits(100000)  # Increase the limit to 10000 digits

react_prompt_header = "Here are some previous solutions and the corresponding test results.\n"
react_prompt_starter = "\n\nYour solution:\n"
extra_header = "\n\nName the function answer()"

class Node:
    def __init__(self, solution: str, parent=None, context="", depth=0):
        self.solution = solution
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.context = ""
        self.depth = depth
        self.reflection = ""
        self.test_feedback = ""

    def uct(self, exploration_weight=1.0):
        if self.visits == 0:
            #return float('inf')
            return self.value
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.value)

    def update(self, reward: float):
        self.visits += 1
        self.value += reward
    

def prune_context_blocks(context: str, max_length: int) -> str:
    """Prune the context to fit within the specified max_length by removing entire blocks of content using 'trial' as a delimiter."""
    if len(context) <= max_length:
        return context
    
    # Split by the block delimiter "trial".
    blocks = context.split('Previous Trial')
    
    # Remove the earliest blocks until the context fits within max_length.
    while len('trial'.join(blocks)) > max_length and blocks:
        blocks.pop(0)
    
    return 'trial'.join(blocks)

def gather_context_from_tree(node: Node) -> Tuple[List[str], List[str]]:
    """
    Given a node, walk up its tree and gather the feedback and reflections 
    from each parent node until the root is reached.
    Args:
        node (Node): The node to start gathering context from.
    Returns:
        Tuple[List[str], List[str]]: Two lists containing the accumulated feedback and reflections.
    """
    accumulated_feedback = []
    accumulated_reflection = []
    num_nodes = 0

    while node and num_nodes < 2:
        num_nodes += 1
        if node.test_feedback:
            accumulated_feedback.append(node.test_feedback)
        if node.reflection:
            accumulated_reflection.append(node.reflection)
        node = node.parent

    # Reverse the lists so that the context from the earliest nodes is first
    return accumulated_feedback[::-1], accumulated_reflection[::-1]

def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)

def run_lats(
    model_name: str,
    language: str,
    max_iters: int,
    verbose: bool,
    instruction: str = "Write some code to print Hello World in Python",
    n_samples: int = 3,
    depth: int = 5,
) -> None:
    exe = executor_factory(language)
    gen = generator_factory(language)
    model = model_factory(model_name)


    num_success = 0  # Counter for successful solutions
    cur_func_impl = None

    item = {}

    #for idx, item in enumerate(dataset):
    
    tests = gen.internal_tests(instruction + extra_header, model, 1)
    tests_i = sample_n_random(tests, 1)

    while cur_func_impl is None:
        cur_func_impl = gen.func_impl(instruction + extra_header, model, "simple")
    root = Node(cur_func_impl) # initial solution (for pass@1 metric)
    
    # Lists for logging
    reflections = []
    implementations = []
    test_feedback = []
    is_solved = False

    # first attempt
    
    implementations.append(cur_func_impl)
    assert isinstance(cur_func_impl, str)
    is_passing, feedback, _ = exe.execute(cur_func_impl)
    test_feedback.append(feedback)

    # if solved, exit early
    if is_passing:
        num_success += 1
        return cur_func_impl # GET SOLUTION
    
    reflection = gen.self_reflection(cur_func_impl, feedback, model)
    reflections += [reflection]
    root.test_feedback = feedback
    root.reflection = reflection
    max_iters = int(max_iters)
    for cur_iter in range(max_iters):
        # Selection
        tests_i = sample_n_random(tests, 1)

        node = root
        trajectory = {
            'solutions': [],
            'feedbacks': []
        }

        while node.children:
            node = node.best_child()
            trajectory['solutions'].append(node.solution)
        
        # Expansion
        for _ in range(n_samples):
            new_solution = None
            strategy = "mcts"
            prev_func_impl = node.solution
            feedback = node.test_feedback
            reflection = node.reflection
            acc_feedback, acc_reflection = gather_context_from_tree(node)
            
            while new_solution is None:
                new_solution = gen.func_impl(
                    func_sig=instruction+extra_header,
                    model=model,
                    strategy=strategy,
                    prev_func_impl=prev_func_impl,
                    feedback=feedback,
                    self_reflection=reflection,
                    acc_feedback = acc_feedback,
                    acc_reflection = acc_reflection
                )

            combined_context = "\nPrevious Trial\n\n" + new_solution

            child = Node(new_solution, parent=node, context=combined_context, depth=node.depth + 1)
            node.children.append(child)

            # Simulation
            reward_real = 0
            for child in node.children:
                is_passing_internal, feedback_internal, _ = exe.execute(child.solution, tests_i)
                if not is_passing_internal:
                    reflection = gen.self_reflection(child.solution, feedback_internal, model)
                    reflections.append(reflection)
                    child.reflection = reflection
                    child.test_feedback = feedback_internal
                    child.context += "\n\nPrevious Trial\n\n" + child.solution + "\n\nTest results: \n" + feedback_internal + "\n\nSelf-reflection: " + reflection
                else:
                    child.context += "\n\nPrevious Trial\n\n" + child.solution + "\n\nTest results: \n" + feedback_internal
                    child.reflection = ""
                    child.test_feedback = feedback_internal

                if "Tested passed:" in feedback_internal:
                    # Split at "Tests failed:" and get the part before it (which contains the passed tests)
                    passed_section = feedback_internal.split("Tests failed:")[0]
                    # Split at "Tested passed:" and get the part after it, then count the non-empty lines
                    reward_internal = len([line for line in passed_section.split("Tested passed:")[1].splitlines() if line.strip() != ''])
                    reward_internal = reward_internal / len(tests_i)
                else:
                    reward_internal = 0
                if is_passing_internal or cur_iter == max_iters - 1:
                    item["solution"] = child.solution
                    break

            if is_solved:
                break
            
            reward = reward_internal + reward_real
            child.update(reward)

            # Backpropagation
            temp = child
            while temp.parent:
                temp = temp.parent
                temp.update(reward)
    
    # Choose the best solution after all iterations
    if is_solved:
        best_solution = item["solution"]
    else:
        best_solution = root.best_child_value().solution
        item["solution"] = best_solution

    return best_solution