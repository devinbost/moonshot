import os

from Chatbot import Chatbot
from Crawler import Crawler
from DataAccess import DataAccess
from UserInterface import UserInterface


class Main:
    def __init__(self):
        self.isFirstRun = True
        self.dataAccess = DataAccess()
        self.crawler = Crawler(self.dataAccess)

    def firstRun(self):
        if self.isFirstRun is True:
            self.crawler.async_crawl_and_ingest(
                "https://dataplatform.cloud.ibm.com/docs/sitemap.xml"
            )
            self.isFirstRun = False

    def startChatbot(self):
        bot = Chatbot(self.dataAccess)
        return bot

    def buildUI(self):
        bot = self.startChatbot()
        ui = UserInterface(self.dataAccess, bot, self.crawler)
        ui.render()


if __name__ == "__main__":
    main = Main()
    main.buildUI()
