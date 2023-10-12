import os
import streamlit as st
from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
from UserInterface import UserInterface


class Main:
    def __init__(self):
        print("running __init__ in Main")
        self.isFirstRun = True
        self.dataAccess: DataAccess = DataAccess()
        self.crawler: Crawler = Crawler(self.dataAccess)
        self.chatbot: Chatbot = None

    def firstRun(self):
        if self.isFirstRun is True:
            self.crawler.async_crawl_and_ingest(
                "https://dataplatform.cloud.ibm.com/docs/sitemap.xml"
            )
            self.isFirstRun = False

    def startChatbot(self):
        print("ran startChatbot")
        if self.chatbot is None:
            self.chatbot = Chatbot(self.dataAccess)

    def buildUI(self):
        print("ran buildUI")
        self.startChatbot()
        ui = UserInterface(self.dataAccess, self.chatbot, self.crawler)
        ui.render()


if __name__ == "__main__":
    print('running __name__ == "__main__"')
    if "main" not in st.session_state:
        print("'main' not in st.session_state")
        main = Main()
        st.session_state["main"] = main
    else:
        print("'main' in st.session_state")
        main = st.session_state["main"]
    main.buildUI()
