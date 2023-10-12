import streamlit as st

from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
import time


class UserInterface:
    def __init__(self, data_access: DataAccess, chatbot: Chatbot, crawler: Crawler):
        print("running __init__ on UserInterface")
        self.app_name = "Chatbot demo"
        self.data_access = data_access
        self.chatbot = chatbot
        self.crawler = crawler

    def render(self):
        print("Running render")
        st.title(self.app_name)

        question = st.text_input("Ask a question for the chatbot")
        searched = st.button("Search")
        if len(question) > 0 and searched:
            bot_response = self.chatbot.runInference(question)
            if bot_response:
                st.write(bot_response["answer"])
                st.write("\n\n\n\n")
                st.write(bot_response)

        if st.button("Crawl LDS docs"):
            progress_bar = st.progress(0, "Percentage completion of site crawling")
            start = time.time()
            sitemapList = [
                "https://sitemaps.churchofjesuschrist.org/sitemap-service/www.churchofjesuschrist.org/en/sitemap_1.xml",
                "https://sitemaps.churchofjesuschrist.org/sitemap-service/www.churchofjesuschrist.org/en/sitemap_2.xml",
                "https://sitemaps.churchofjesuschrist.org/sitemap-service/www.churchofjesuschrist.org/en/sitemap_3.xml",
                "https://sitemaps.churchofjesuschrist.org/sitemap-service/www.churchofjesuschrist.org/en/sitemap_4.xml",
                "https://sitemaps.churchofjesuschrist.org/sitemap-service/www.churchofjesuschrist.org/en/sitemap_5.xml",
            ]
            self.crawler.async_crawl_and_ingest_list(sitemapList, progress_bar)
            end = time.time()
            completionTime = end - start  # Time elapsed in seconds
            st.caption(f"Completed parsing IBM docs in {completionTime} seconds")
        if st.button("Crawl IBM docs"):
            progress_bar = st.progress(0, "Percentage completion of site crawling")
            start = time.time()
            self.crawler.async_crawl_and_ingest(
                "https://dataplatform.cloud.ibm.com/docs/sitemap.xml", progress_bar
            )
            end = time.time()
            completionTime = end - start  # Time elapsed in seconds
            st.caption(f"Completed parsing IBM docs in {completionTime} seconds")
        if st.button("Clear dataset"):
            self.data_access.vector_store.clear()
        if st.button("Load Wikipedia dataset"):
            self.data_access.loadWikipediaData()
