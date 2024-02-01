import os
import streamlit as st
from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
from SitemapCrawler import SitemapCrawler
from UserInterface import UserInterface, render_new
from core.ConfigLoader import ConfigLoader
from core.EmbeddingManager import EmbeddingManager
from core.VectorStoreFactory import VectorStoreFactory


class Main:
    def __init__(self):
        print("running __init__ in Main")
        embedding_manager = EmbeddingManager()
        config_loader = ConfigLoader()
        vector_store_factory = VectorStoreFactory(embedding_manager, config_loader)
        self.data_access: DataAccess = DataAccess(
            config_loader, embedding_manager, vector_store_factory
        )
        self.crawler: Crawler = Crawler(self.data_access)
        self.chatbot: Chatbot = None
        self.sitemap_crawler: SitemapCrawler = SitemapCrawler()

    def startChatbot(self):
        print("ran startChatbot")
        if self.chatbot is None:
            self.chatbot = Chatbot(self.data_access)

    def buildUI(self):
        print("ran buildUI")
        self.startChatbot()
        ui = UserInterface(self.chatbot, self.crawler, self.sitemap_crawler)
        render_new(self.data_access, ui.chatbot, ui.crawler)


def main():
    if "main" not in st.session_state:
        st.session_state.main = Main()

    st.session_state.main.buildUI()


if __name__ == "__main__":
    print('running __name__ == "__main__"')
    st.set_page_config(layout="wide")
    main()
