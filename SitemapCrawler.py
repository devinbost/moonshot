import asyncio
import logging

import os
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from DataAccess import DataAccess

import config

# Configure logging
logging.basicConfig(
    filename=config.scratch_path + "/crawler.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


class SitemapCrawler:
    def __init__(self):
        self.crawled_urls = None
        self.sitemap_gen_semaphore = asyncio.Semaphore(20)

    async def fetch(self, url, session):
        async with self.sitemap_gen_semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                print(f"Error: {e}")
                return ""

    async def get_all_links(self, url, session):
        content = await self.fetch(url, session)
        soup = BeautifulSoup(content, "html.parser")
        links = [
            urljoin(url, link.get("href")) for link in soup.find_all("a", href=True)
        ]
        return set(links)

    async def crawl_website(self, base_url, max_depth=2, depth=0):
        if depth > max_depth:
            return
        async with aiohttp.ClientSession() as session:
            links = await self.get_all_links(base_url, session)
            for link in links:
                if (
                    link not in self.crawled_urls
                    and urlparse(link).netloc == urlparse(base_url).netloc
                ):
                    print(f"Crawling: {link}")
                    self.crawled_urls.add(link)
                    await self.crawl_website(link, max_depth, depth + 1)

    def start_sitemap_building(self, base_url):
        crawled_urls = set()

        base_url = os.getenv("CRAWLER_BASE_URL" "https://www.kubeflow.org")
        url_file = scratch_path + "/urls.txt"

        # Run asynchronous crawl
        asyncio.run(self.crawl_website(base_url))

        # Write URLs to a text file
        with open(url_file, "w") as file:
            for url in crawled_urls:
                file.write(url + "\n")

        print(f"URLs written to {url_file} successfully.")
