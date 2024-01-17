#First pass at a function to build a linked representation of the memory

def convert_to_graphql_v3(data):
    sorted_data = sorted(data, key=lambda x: float(x["step"]))
    
    graphql_format = "{\n  steps {"
    last_main_step = None

    for entry in sorted_data:
        step_id = float(entry["step"])
        user_goal = entry["user_goal"]
        solution = entry["solution"]
        output = entry["output"]
        
        if step_id.is_integer():
            linked_to = last_main_step
            last_main_step = step_id
        else:
            linked_to = int(step_id)
        
        linked_to_str = f'\n    linkedTo: {linked_to if linked_to is not None else ""}'
        
        graphql_format += f'''
            id: {step_id}
            userGoal: "{user_goal}"
            solution: "{solution.strip()}"
            output: "{output if isinstance(output, str) else ''}"{linked_to_str}
        '''

    graphql_format += "\n  }\n}"

    return graphql_format