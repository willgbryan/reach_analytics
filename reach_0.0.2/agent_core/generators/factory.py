from .py_generate import PyGenerator
from .generator_types import Generator
from .model import ModelBase, GPT4, GPT35, GPTDavinci

def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo-0613":
        return GPT35()
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")