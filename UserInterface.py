import streamlit as st
from datetime import datetime
from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
from SitemapCrawler import SitemapCrawler
from graphviz import Digraph


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
        if "data_map" not in st.session_state:
            st.session_state["data_map"] = {}
        if "graph" not in st.session_state:
            st.session_state["graph"] = Digraph(comment="Component Graph")

    def generate_unique_id(self):
        return str(datetime.utcnow())  # "test"  # str(uuid.uuid4())

    def get_param_inputs(self):
        NotImplementedError()

    def get_output_vars(self):
        return {
            form_data.output_var
            for form_data in st.session_state["data_map"].values()
            if form_data.output_var
        }
