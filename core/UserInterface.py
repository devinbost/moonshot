import asyncio

import streamlit as st
from datetime import datetime
import time
from Crawler import Crawler
from SitemapCrawler import SitemapCrawler
from graphviz import Digraph
from sortedcontainers import SortedSet

from Chatbot import Chatbot
from ClassInspector import (
    get_importable_classes,
    get_class_method_params,
    get_class_attributes_from_llm,
)
from core.CollectionManager import CollectionManager
from core.ConfigLoader import ConfigLoader
from core.EmbeddingManager import EmbeddingManager
from core.SplitterFactory import SplitterFactory
from core.VectorStoreFactory import VectorStoreFactory
from pydantic_models.ComponentData import ComponentData
from DataAccess import DataAccess
from pydantic_models.PropertyInfo import PropertyInfo
from pydantic_models.TableDescription import TableDescription
from pydantic_models.UserInfo import UserInfo


class UserInterface:
    def __init__(
        self,
        chatbot: Chatbot,
        crawler: Crawler,
        sitemap_crawler: SitemapCrawler,
    ):
        print("running __init__ on UserInterface")
        self.app_name = "Chatbot demo"
        self.chatbot = chatbot
        self.crawler = crawler
        self.sitemap_crawler = sitemap_crawler
        embedding_manager = EmbeddingManager()
        config_loader = ConfigLoader()
        self.vector_store_factory = VectorStoreFactory(embedding_manager, config_loader)
        self.astrapy_db = self.vector_store_factory.create_vector_store("AstraPyDB")
        self.collection_manager = CollectionManager(self.astrapy_db, embedding_manager, "sitemapls")

    def setup_prompt_ui_components(self, column):
        prompt_search = column.text_input("Find prompt")
        matching_prompts = self.collection_manager.get_matching_prompts(prompt_search)
        prompts = [s["prompt"] for s in matching_prompts]
        question_list = column.selectbox("Select existing prompt", SortedSet(prompts))
        load_prompt = column.button("Load prompt")
        if load_prompt:
            question = column.text_area(
                "Ask a question for the chatbot", value=question_list
            )
        else:
            question = column.text_area("Ask a question for the chatbot")
        save_button = column.button("Save prompt")
        if save_button:
            self.collection_manager.save_prompt(question)
        return question


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
):
    # Extract values from the UI components
    # Assuming these values are stored in variables like self.import_path, self.stage, etc.
    # For the 'new' case in selectboxes, replace with text_input values
    component_params = {}
    component_name = ""
    component_type = ""
    if access_type == "method":
        method_params_processed = {
            k: (method_param_inputs[k] if v == "new" else v)
            for k, v in method_param_dropdowns.items()
            if method_param_inputs.get(k, v) not in (None, "")
        }
        component_params = method_params_processed
        component_name = selected_method
        if st.checkbox("Inference method", value=False):
            component_type = "inference"
        else:
            component_type = "setup"
    elif access_type == "constructor":
        constructor_params_processed = {
            k: (constructor_param_inputs[k] if v == "new" else v)
            for k, v in constructor_param_dropdowns.items()
            if constructor_param_inputs.get(k, v) not in (None, "")
        }
        component_params = constructor_params_processed
        component_name = "constructor"  # note this.
        component_type = "setup"
    elif access_type == "property":
        NotImplementedError('elif access_type == "property":')
        component_type = "setup"
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


def build_reflection_menu(data_access, col1):
    # action_type = col1.selectbox("Action type", ("import", "env", "secret"))
    library = col1.selectbox(
        "Import path",
        ("langchain", "langchain_core", "langchain_community", "llamaindex", "astrapy"),
    )
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
                    "agents",
                    "chains",
                    "embeddings",
                    "llms",
                    "document_loaders",
                ),
            )
            # Need to add more
            matching_classes = [s for s in class_names if class_type in s]
            class_name = col1.selectbox("Class name", SortedSet(matching_classes))
        else:
            class_name = col1.selectbox("Class name", SortedSet(class_names))
        if class_name is not None:  # i.e. if the filter returned results, if applicable
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


def setup_sitemap_crawler_ui(column, crawler: Crawler, collection_name: str, chunk_size: int, chunk_overlap: int):
    sitemaps = column.text_input(
        "Sitemap URLs to crawl as csv"
    )  # To do: Handle this input better
    if column.button("Crawl docs"):
        progress_bar = column.progress(0, "Percentage completion of site crawling")
        start = time.time()
        # Check if empty
        sitemap_list = get_sitemap_list_from_csv(sitemaps)
        embedding_manager = EmbeddingManager()
        config_loader = ConfigLoader()
        # I could update the ConfigLoader to first load from a YAML config file.
        vector_store_factory = VectorStoreFactory(embedding_manager, config_loader)
        vector_store = vector_store_factory.create_vector_store(
            "AstraDB", collection_name=collection_name
        )
        splitter_factory = SplitterFactory(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitter = splitter_factory.create_splitter()
        crawler.async_crawl_and_ingest_list(
            sitemap_list, progress_bar, vector_store, splitter
        )
        end = time.time()
        completion_time = end - start  # Time elapsed in seconds
        column.caption(f"Completed parsing docs in {completion_time} seconds")


def render_new(data_access: DataAccess, chatbot: Chatbot, crawler: Crawler):
    col1, col2 = st.columns(2)
    if col1.checkbox("Preview Mode?", value=False):
        build_reflection_menu(data_access, col1)
        build_graph_display(col2, data_access)

    user_info: UserInfo = UserInfo(
        properties=[
            PropertyInfo(property_name="age", property_type="int", property_value=30),
            PropertyInfo(
                property_name="name",
                property_type="text",
                property_value="John Smith",
            ),
            PropertyInfo(
                property_name="phone_number",
                property_type="text",
                property_value="555-555-5555",
            ),
            PropertyInfo(
                property_name="email",
                property_type="text",
                property_value="johndoe@example.com",
            ),
            PropertyInfo(
                property_name="address",
                property_type="text",
                property_value="123 Main St, Anytown, USA",
            ),
            PropertyInfo(
                property_name="account_status",
                property_type="text",
                property_value="Active",
            ),
            PropertyInfo(
                property_name="plan_type",
                property_type="text",
                property_value="Unlimited Data Plan",
            ),
        ]
    )

    # Defaults:
    collection_name = "sitemapls"
    chunk_size = "300"
    chunk_overlap = "150"
    embedding_model = "all-MiniLM-L12-v2"

    # Configurations UI section (for config overrides)
    if col1.checkbox("Show configs", value=False):
        # (Update values based on user input)
        collection_name = col1.text_input("Name of KB collection name", collection_name)
        chunk_size = col1.text_input("Chunk size", chunk_size)
        chunk_overlap = col1.text_input("Chunk overlap", chunk_overlap)
        embedding_model = col1.text_input("Huggingface embedding model to use", embedding_model)

    if col1.checkbox("Enable web crawler?", value=False):
        setup_sitemap_crawler_ui(col2, crawler, collection_name, int(chunk_size), int(chunk_overlap))
    user_chat_area = col1.text_area("Enter message here")
    question_count = col1.text_input("Number of questions to ask for each table", "3")

    searched = col1.button("Search")

    summarization_limit = 3  # We can make this a config param

    if len(user_chat_area) > 0 and searched:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                chatbot.answer_customer(
                    user_chat_area, user_info, col2, col1, summarization_limit, question_count, collection_name,
                    embedding_model
                )
            )
        finally:
            loop.close()

        # bot_response = chatbot.answer_customer(user_chat_area, user_info, col2)
        # bot_chat_area = col1.markdown(bot_response)


def build_graph_display(col2, data_access):
    # Build graph from data structure here:
    graph = Digraph(comment="Component Graph")
    data_map = data_access.get_data_map()
    updated_graph = data_access.build_graph(data_map, graph)
    col2.graphviz_chart(updated_graph)
    python_code = data_access.generate_python_code()
    col2.markdown(python_code)


def get_sitemap_list_from_csv(sitemaps):
    if sitemaps.strip():
        # Split the string into a list and strip any extra spaces
        cleaned_list = [item.strip() for item in sitemaps.split(",")]
    else:
        # If the string is empty, return an empty list
        cleaned_list = []
    return cleaned_list
