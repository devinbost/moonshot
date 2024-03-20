import os
import streamlit as st
from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
from SitemapCrawler import SitemapCrawler
from UserInterface import UserInterface
from core.ConfigLoader import ConfigLoader
from core.VectorStoreFactory import VectorStoreFactory


class Main:
    def __init__(self):
        print("running __init__ in Main")
        self.crawler: Crawler = Crawler()
        self.sitemap_crawler: SitemapCrawler = SitemapCrawler()

    def buildUI(self):
        print("ran buildUI")
        ui = UserInterface(self.crawler, self.sitemap_crawler)
        ui.render_new(ui.crawler)


def main():
    if "main" not in st.session_state:
        st.session_state.main = Main()

    st.session_state.main.buildUI()


if __name__ == "__main__":
    print('running __name__ == "__main__"')
    st.set_page_config(layout="wide")
    main()
