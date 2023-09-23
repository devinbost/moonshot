import streamlit as st

from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess


class UserInterface:
    def __init__(self, data_access: DataAccess, chatbot: Chatbot, crawler: Crawler):
        self.app_name = "IBM Watson demo"
        self.data_access = data_access
        self.chatbot = chatbot
        self.crawler = crawler

    def render(self):
        st.title(self.app_name)

        question = st.text_input("Ask a question for the chatbot")
        if len(question) > 0:
            answer = self.chatbot.runInference(question)
            if answer:
                st.write(answer)
        if st.button("Clear dataset"):
            self.data_access.vector_store.clear()
        if st.button("Load Wikipedia dataset"):
            self.data_access.loadWikipediaData()
        if st.button("Crawl IBM docs"):
            progress_bar = st.progress(0, "Percentage completion of site crawling")
            self.crawler.async_crawl_and_ingest(
                "https://dataplatform.cloud.ibm.com/docs/sitemap.xml", progress_bar
            )
