import enum
import importlib
import inspect
import json
import pkgutil

from langchain import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import logging
import json


def get_class_attributes(cls):
    # Filter to get only attributes with type annotations
    attributes = {
        name: attr
        for name, attr in inspect.getmembers(cls)
        if not name.startswith("_") and not inspect.isroutine(attr)
    }

    # Optionally, filter out attributes without type annotations
    # attributes = {name: attr for name, attr in attributes.items() if hasattr(attr, '__annotations__')}

    return attributes


def get_class_attributes_from_llm(cls):
    prompt_template = """You're a helpful assistant. When given the following Python source code, give me the attributes 
mentioned in the provided code and comments (if any) as a JSON object with the name, type, and default (if any) of each 
attribute. For example, from the following CODE (including comments), return the JSON provided. Don't provide any explanation. Format the results as JSON only so that I can parse it with the json.loads method in Python.
Also, prefer using defaults in the comments over the code. For example, if a comment specifies model_name = "sentence-transformers/all-mpnet-base-v2" but the code later specifies model_name: str = DEFAULT_MODEL_NAME, 
then the type is str, but the default is "sentence-transformers/all-mpnet-base-v2".
Additionally, if a comment specifies model_kwargs = {{'device': 'cpu'}} but the code shows model_kwargs: Dict[str, Any] = Field(default_factory=dict)
then the type is Dict[str, Any] but the default is {{'device': 'cpu'}}.
If no default is available for an attribute, set it to "None".
Also, give the outputs without backticks or markdown. I just want raw text.

EXAMPLE CODE:

class HuggingFaceEmbeddings(BaseModel, Embeddings):
\"\"\"HuggingFace sentence_transformers embedding models.

To use, you should have the ``sentence_transformers`` python package installed.

Example:
    .. code-block:: python

        from langchain.embeddings import HuggingFaceEmbeddings

        model_name = \"sentence-transformers/all-mpnet-base-v2\"
        model_kwargs = {{'device': 'cpu'}}
        encode_kwargs = {{'normalize_embeddings': False}}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
\"\"\"

client: Any  #: :meta private:
model_name: str = DEFAULT_MODEL_NAME
\"\"\"Model name to use.\"\"\"
cache_folder: Optional[str] = None
\"\"\"Path to store models. 
Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.\"\"\"
model_kwargs: Dict[str, Any] = Field(default_factory=dict)
\"\"\"Key word arguments to pass to the model.\"\"\"
encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
\"\"\"Key word arguments to pass when calling the `encode` method of the model.\"\"\"
multi_process: bool = False
\"\"\"Run encode() on multiple GPUs.\"\"\"

def __init__(self, **kwargs: Any):
    \"\"\"Initialize the sentence_transformer.\"\"\"
    super().__init__(**kwargs)
    try:
        import sentence_transformers

    except ImportError as exc:
        raise ImportError(
            \"Could not import sentence_transformers python package. "
            "Please install it with `pip install sentence_transformers`.\"
        ) from exc

    self.client = sentence_transformers.SentenceTransformer(
        self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
    )

class Config:
    \"\"\"Configuration for this pydantic object.\"\"\"

    extra = Extra.forbid

def embed_documents(self, texts: List[str]) -> List[List[float]]:
    \"\"\"Compute doc embeddings using a HuggingFace transformer model.

    Args:
        texts: The list of texts to embed.

    Returns:
        List of embeddings, one for each text.
    \"\"\"
    import sentence_transformers

    texts = list(map(lambda x: x.replace("\n", " "), texts))
    if self.multi_process:
        pool = self.client.start_multi_process_pool()
        embeddings = self.client.encode_multi_process(texts, pool)
        sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
    else:
        embeddings = self.client.encode(texts, **self.encode_kwargs)

    return embeddings.tolist()

def embed_query(self, text: str) -> List[float]:
    \"\"\"Compute query embeddings using a HuggingFace transformer model.

    Args:
        text: The text to embed.

    Returns:
        Embeddings for the text.
    \"\"\"
    return self.embed_documents([text])[0]

EXAMPLE JSON:


[
    {{
        \"name\": \"client\",
        \"type\": \"Any\",
        \"default\": \"None\"
    }},
    {{
        \"name\": \"model_name\",
        \"type\": \"str\",
        \"default\": \"sentence-transformers/all-mpnet-base-v2\"
    }},
    {{
        \"name\": \"cache_folder\",
        \"type\": \"Optional[str]\",
        \"default\": \"None\"
    }},
    {{
        \"name\": \"model_kwargs\",
        \"type\": \"Dict[str, Any]\",
        \"default\": \"{{'device': 'cpu'}}\"
    }},
    {{
        \"name\": \"encode_kwargs\",
        \"type\": \"Dict[str, Any]\",
        \"default\": \"{{'normalize_embeddings': False}}\"
    }},
    {{
        \"name\": \"multi_process\",
        \"type\": \"bool\",
        \"default\": \"False\"
    }}
]

ACTUAL CODE:
{source_code}

ACTUAL JSON:
            """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = OpenAI(model_name="gpt-4-1106-preview")  # "gpt-3.5-turbo-1106")
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    source_code = inspect.getsource(cls)
    result = chain.invoke({"source_code": source_code})

    cleaned_result = remove_json_formatting(result)
    logging.info(f"Got result from LLM on class attributes: {cleaned_result}")

    json_result = json.loads(cleaned_result)

    return json_result


def remove_json_formatting(input_string: str):
    # Check if the string starts with ```json and ends with ```
    if input_string.startswith("```json") and input_string.endswith("```"):
        # Remove the ```json at the start and ``` at the end
        return input_string[len("```json") : -len("```")].strip()
    else:
        # Return the original string if it doesn't start with ```json and end with ```
        return input_string


def get_class_method_params(cls):
    # Dictionary to store method names and their parameters
    methods_with_params = {}

    # Iterate over all members of the class
    for name, member in inspect.getmembers(cls):
        if not name.startswith("_"):
            # Check if the member is a method
            if inspect.isfunction(member) or inspect.ismethod(member):
                # Get the signature of the method
                signature = inspect.signature(member)
                # Store the method name and its parameters
                methods_with_params[name] = [
                    param.name
                    for param in signature.parameters.values()
                    if param.name != "self"
                ]
    return methods_with_params


def get_importable_classes(module_name):
    def walk_modules(module):
        classes = {}
        for loader, modname, ispkg in pkgutil.walk_packages(
            module.__path__, prefix=module.__name__ + "."
        ):
            try:
                sub_module = importlib.import_module(modname)
                if ispkg:
                    classes.extend(walk_modules(sub_module))
                else:
                    for name, obj in inspect.getmembers(sub_module, inspect.isclass):
                        if obj.__module__ == sub_module.__name__ and not issubclass(
                            obj, enum.Enum
                        ):
                            classes[f"{obj.__module__}.{obj.__name__}"] = obj
            except Exception as e:
                print(f"Error importing module {modname}: {e}")
                continue
        return classes

    try:
        main_module = importlib.import_module(module_name)
        return walk_modules(main_module)
    except Exception as e:
        print(f"Error importing main module {module_name}: {e}")
        return []
