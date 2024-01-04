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
from pydantic_models.ComponentData import ComponentData
from DataAccess import DataAccess
from pydantic_models.PropertyInfo import PropertyInfo
from pydantic_models.TableDescription import TableDescription
from pydantic_models.UserInfo import UserInfo


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


def setup_sitemap_crawler_ui(column, crawler: Crawler):
    sitemaps = column.text_input(
        "Sitemap URLs to crawl as csv"
    )  # To do: Handle this input better
    table_name = column.text_input("Collection to store data")
    if column.button("Crawl docs"):
        progress_bar = column.progress(0, "Percentage completion of site crawling")
        start = time.time()
        # Check if empty
        sitemap_list = get_sitemap_list_from_csv(sitemaps)
        crawler.async_crawl_and_ingest_list(sitemap_list, progress_bar, table_name)
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
    if col1.checkbox("Enable web crawler?", value=False):
        setup_sitemap_crawler_ui(col2, crawler)
    user_chat_area = col1.text_area("Enter message here")
    searched = col1.button("Search")

    if len(user_chat_area) > 0 and searched:
        bot_response = chatbot.answer_customer(user_chat_area, user_info, col2)
        bot_chat_area = col1.text_area(bot_response)


def render(data_access: DataAccess, app_name, chatbot: Chatbot, crawler):
    col1, col2, col3 = st.columns(3)
    build_reflection_menu(data_access, col1)

    # Build graph from data structure here:
    graph = Digraph(comment="Component Graph")
    data_map = data_access.get_data_map()
    updated_graph = data_access.build_graph(data_map, graph)
    col2.graphviz_chart(updated_graph)
    python_code = data_access.generate_python_code()
    col2.markdown(python_code)
    print("Running render")
    col3.title(app_name)

    question = setup_prompt_ui_components(col3, data_access)
    ann_terms = col3.text_input("Terms for ANN")
    searched = col3.button("Search")
    if len(question) > 0 and searched:
        bot_response = chatbot.run_inference_astrapy(
            terms_for_ann=ann_terms,
            ann_length=5,
            collection="langchain_demo1",
            question=question,
        )
        if bot_response:
            col3.write(bot_response)
    search_via_query = col3.button("Query")
    desc1 = TableDescription(
        table_name="support_cases",
        column_name="phonenumber",
        description="For user's cell phone, which is a user ID",
    )
    desc2 = TableDescription(
        table_name="support_cases",
        column_name="case_transcript",
        description="Contains support transcript for a given case / ticket",
    )
    desc3 = TableDescription(
        table_name="user_plan",
        column_name="phonenumber",
        description="For user's cell phone, which is a user ID",
    )
    desc4 = TableDescription(
        table_name="user_plan",
        column_name="plan_details",
        description="Describes the plan information",
    )
    table_descriptions = [desc1, desc2, desc3, desc4]

    if search_via_query:
        data_access.summarize_relevant_tables(
            user_messages=f"USER QUESTION: {question} \n"
            + " \n \n USER CONTEXT: {userID: 1354, phone: 'galaxy s22', plan: 'unlimited', phonenumber: '238-345-7823'}",
            tables=table_descriptions,
        )
    sitemaps = col3.text_input(
        "Sitemap URLs to crawl as csv"
    )  # To do: Handle this input better
    table_name = col3.text_input("Collection to store data")
    if col3.button("Crawl docs"):
        progress_bar = col3.progress(0, "Percentage completion of site crawling")
        start = time.time()
        # Check if empty
        sitemap_list = get_sitemap_list_from_csv(sitemaps)
        crawler.async_crawl_and_ingest_list(sitemap_list, progress_bar, table_name)
        end = time.time()
        completion_time = end - start  # Time elapsed in seconds
        col3.caption(f"Completed parsing docs in {completion_time} seconds")


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


def setup_prompt_ui_components(column, data_access):
    prompt_search = column.text_input("Find prompt")
    matching_prompts = data_access.get_matching_prompts(prompt_search)
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
        data_access.save_prompt(question)
    return question
