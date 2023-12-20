import streamlit as st
from datetime import datetime
from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
from SitemapCrawler import SitemapCrawler
from graphviz import Digraph
from sortedcontainers import SortedSet

from Chatbot import Chatbot
from ClassInspector import (
    get_importable_classes,
    get_class_method_params,
    get_class_attributes_from_llm,
)
from ComponentData import ComponentData
from DataAccess import DataAccess


class UserInterface:
    def __init__(
        self,
        data_access: DataAccess,
        chatbot: Chatbot,
        crawler: Crawler,
        sitemap_crawler: SitemapCrawler,
    ):
        print("running __init__ on UserInterface")
        self.app_name = "Chatbot demo"
        self.data_access = data_access
        self.chatbot = chatbot
        self.crawler = crawler
        self.sitemap_crawler = sitemap_crawler

    def generate_unique_id(self):
        return str(datetime.utcnow())  # "test"  # str(uuid.uuid4())

    def get_param_inputs(self):
        NotImplementedError()


def build_param_inputs(
    data_access, col1, param_dropdowns, param_inputs, param_names, prefix: str
):
    for param in param_names:
        container = col1.container(border=True)
        param_dropdowns[param] = container.selectbox(
            f"{prefix} parameter: {param}",
            data_access.get_output_variable_names(),
        )
        if param_dropdowns[param] == "new":
            param_inputs[param] = container.text_input(
                f":orange Enter value for: {param}"
            )


def build_llm_param_inputs(
    data_access, col1, param_dropdowns, param_inputs, param_names, prefix: str
):
    for param in param_names:
        container = col1.container(border=True)
        param_name = param["name"]
        param_type = param["type"]
        param_default = param["default"]
        param_dropdowns[param_name] = container.selectbox(
            f"{prefix} parameter: {param_name}",
            data_access.get_output_variable_names(),
        )
        if param_dropdowns[param_name] == "new":
            param_inputs[param_name] = container.text_input(
                f":orange Enter value for: {param_name} | Type: {param_type}",
                value=param_default,
            )


def add_element(
    library,
    stage,
    class_name,
    access_type,
    selected_method,
    method_param_dropdowns,
    method_param_inputs,
    constructor_param_dropdowns,
    constructor_param_inputs,
    output_var,
    component_type=None,
):
    # Extract values from the UI components
    # Assuming these values are stored in variables like self.import_path, self.stage, etc.
    # For the 'new' case in selectboxes, replace with text_input values
    component_params = {}
    component_name = ""
    if access_type == "method":
        method_params_processed = {
            k: (method_param_inputs[k] if v == "new" else v)
            for k, v in method_param_dropdowns.items()
            if method_param_inputs.get(k, v) not in (None, "")
        }
        component_params = method_params_processed
        component_name = selected_method
    elif access_type == "constructor":
        constructor_params_processed = {
            k: (constructor_param_inputs[k] if v == "new" else v)
            for k, v in constructor_param_dropdowns.items()
            if constructor_param_inputs.get(k, v) not in (None, "")
        }
        component_params = constructor_params_processed
        component_name = "constructor"  # note this.
    elif access_type == "property":
        NotImplementedError('elif access_type == "property":')
    else:
        NotImplementedError(
            "must currently use method, constructor, or property for access_type"
        )
    component_id = f"{class_name}-{component_name}"  # self.generate_unique_id()
    # Create a FormData object
    form_data = ComponentData(
        id=component_id,
        library=library,
        class_name=class_name,
        component_name=component_name,
        access_type=access_type,
        params=component_params,
        output_var=output_var,
        component_type=component_type,
    )
    return form_data


def buildInputs(data_access, col1):
    # action_type = col1.selectbox("Action type", ("import", "env", "secret"))
    library = col1.selectbox("Import path", ("langchain", "llamaindex", "astrapy"))
    stage = col1.selectbox("Stage", ("template", "created"))
    class_name = access_type = method_param_dropdowns = method_param_inputs = None
    constructor_param_dropdowns = (
        selected_method
    ) = constructor_param_inputs = output_var = None
    # Perhaps we need a mechanism for handling order for created objects in case execution order ever matters still after DAG is resolved.
    if stage == "template":
        classes = get_importable_classes(library)
        class_names = list(classes.keys())
        if col1.checkbox("Filtered", value=True):
            class_type = col1.selectbox(
                "Class type",
                (
                    "agents.agent",
                    "langchain.chains",
                    "embeddings",
                    "langchain.llms",
                    "document_loaders",
                ),
            )
            # Need to add more
            matching_classes = [s for s in class_names if class_type in s]
            class_name = col1.selectbox("Class name", SortedSet(matching_classes))
        else:
            class_name = col1.selectbox("Class name", SortedSet(class_names))
        class_obj = classes[class_name]
        access_type = col1.selectbox(
            "Access type", ("property", "method", "constructor")
        )
        if access_type == "method":
            methods = get_class_method_params(class_obj)
            method_names = methods.keys()
            selected_method = col1.selectbox("Method name", SortedSet(method_names))
            param_names = methods[selected_method]
            method_param_inputs = {}
            method_param_dropdowns = {}

            build_param_inputs(
                data_access,
                col1,
                method_param_dropdowns,
                method_param_inputs,
                param_names,
                "Method",
            )
        elif access_type == "property":
            NotImplementedError("Needs to implement property get/set actions")
        elif access_type == "constructor":
            param_names = get_class_attributes_from_llm(
                class_obj
            )  # self.get_class_attributes(class_obj)
            constructor_param_inputs = {}
            constructor_param_dropdowns = {}

            # It would be nice to find some way to mark which params are required.

            # Additionally, at some point, we probably need a layer to simplify names for business people.
            # And, we need to make it so that environment variables and secrets can be referenced after being added from another place.
            build_llm_param_inputs(
                data_access,
                col1,
                constructor_param_dropdowns,
                constructor_param_inputs,
                param_names,
                "Constructor",
            )
        output_var = col1.text_input("Output variable name")
    addButton = col1.button("Add")
    # Add validation here to ensure we're not adding junk in required inputs are blank.
    if addButton:
        form_data = add_element(
            library=library,
            stage=stage,
            class_name=class_name,
            access_type=access_type,
            selected_method=selected_method,
            method_param_dropdowns=method_param_dropdowns,
            method_param_inputs=method_param_inputs,
            constructor_param_dropdowns=constructor_param_dropdowns,
            constructor_param_inputs=constructor_param_inputs,
            output_var=output_var,
        )
        data_access.add_component(form_data)
    # component_type = col1.selectbox("Component Type", ("Construction", "Inference"))


def render(data_access: DataAccess, app_name, chatbot: Chatbot, crawler):
    col1, col2, col3 = st.columns(3)
    buildInputs(data_access, col1)

    # Build graph from data structure here:
    graph = Digraph(comment="Component Graph")
    data_map = data_access.get_data_map()
    updated_graph = data_access.build_graph(data_map, graph)
    col2.graphviz_chart(updated_graph)

    print("Running render")
    col3.title(app_name)
    prompt = col3.text_input("Prompt template")
    question = col3.text_input("Ask a question for the chatbot")
    title_filter = col3.text_input("(optional) Title Filter")
    result_size = col3.text_input("(optional) Results to retrieve")
    searched = col3.button("Search")
    if len(question) > 0 and searched:
        bot_response = chatbot.runInference(question, prompt, title_filter, result_size)
        if bot_response:
            col3.write(bot_response["answer"])
            col3.write("\n\n\n\n")
            col3.write(bot_response)
