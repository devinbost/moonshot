import asyncio
import json
import logging
import os
import requests
from typing import List, Optional, Set, Union
import spacy
import aiohttp
import pandas as pd
from astrapy.db import AstraDB as AstraPyDB, AstraDBCollection
from bs4 import BeautifulSoup, NavigableString
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session
from cassandra.query import SimpleStatement, dict_factory
from graphviz import Digraph
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import AstraDB, Cassandra
from pandas._typing import ArrayLike
from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.ComponentData import ComponentData
from pydantic_models.FamilyData import FamilyData
from pydantic_models.Person import Person
from pydantic_models.TableDescription import TableDescription
from pydantic_models.TableKey import TableKey
from pydantic_models.TableSchema import TableSchema
from sentence_transformers import SentenceTransformer
import ClassInspector
from Config import config
import hashlib
import wget
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel
import asyncio
from typing import Optional, List, Union, Set

import requests
from bs4 import BeautifulSoup, NavigableString
from typing import Optional, List


class FamilyCrawler:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    # def consolidate_life_summaries(
    #     self,
    #     family_data_list: Union[List[FamilyData], List[List[FamilyData]]],
    #     seen_summaries: Set[str] = None,
    # ) -> str:
    #     """
    #     Consolidates all unique life_summary values from a list of FamilyData objects into a single string,
    #     with each summary separated by a new line character. Handles nested lists and eliminates duplicates.
    #
    #     Args:
    #     family_data_list (Union[List[FamilyData], List[List[FamilyData]]]): A list of FamilyData objects or nested lists of FamilyData objects.
    #     seen_summaries (Set[str]): A set to keep track of already seen life summaries.
    #
    #     Returns:
    #     str: A string containing all unique life_summaries, separated by new lines.
    #     """
    #     if seen_summaries is None:
    #         seen_summaries = set()
    #
    #     summaries = []
    #
    #     for family in family_data_list:
    #         if (
    #             isinstance(family, FamilyData)
    #             and family.life_summary
    #             and family.life_summary not in seen_summaries
    #         ):
    #             summaries.append(family.life_summary)
    #             seen_summaries.add(family.life_summary)
    #         elif isinstance(family, list):
    #             nested_summaries = self.consolidate_life_summaries(
    #                 family, seen_summaries
    #             )
    #             if nested_summaries:
    #                 summaries.append(nested_summaries)
    #
    #     return "\n".join(summaries)
    #
    # async def extract_deep_extended_family_data(
    #     self, url: str
    # ) -> List[List[FamilyData]]:
    #     """
    #     Recursively extracts extended family data for each family member, going two levels deep.
    #
    #     Args:
    #     url (str): The initial URL to start the extraction from.
    #
    #     Returns:
    #     List[FamilyData]: A list of FamilyData objects for each family member's extended family,
    #                       including the extended family of each member.
    #     """
    #     # First level of family data extraction
    #     first_level_family_data = await self.fetch_individual_family_data(url)
    #     second_level_data = []
    #
    #     # Second level of family data extraction for each family member
    #     for person in first_level_family_data.family_members:
    #         family_next_data: List[
    #             FamilyData
    #         ] = await self.extract_extended_family_data(person.url)
    #         second_level_data.append(family_next_data)
    #
    #     # Combine first level and second level results
    #     all_family_data = [first_level_family_data] + second_level_data
    #     return all_family_data
    #
    # # Usage:
    # # deep_extended_family_data = await extract_deep_extended_family_data("your_initial_url_here")
    #
    # async def fetch_individual_family_data(self, url: str) -> FamilyData:
    #     """
    #     Asynchronously fetches family data for an individual's URL.
    #
    #     Args:
    #     url (str): URL of the individual's family data page.
    #
    #     Returns:
    #     FamilyData: Extracted family data for the individual.
    #     """
    #     return await self.extract_family_data(url)
    #
    # async def extract_extended_family_data(self, url: str) -> List[FamilyData]:
    #     """
    #     Asynchronously extracts extended family data for each family member in the given FamilyData object.
    #
    #     Args:
    #     family_data (FamilyData): The initial family data containing family members' URLs.
    #
    #     Returns:
    #     List[FamilyData]: A list of FamilyData objects for each family member's extended family.
    #     """
    #     first_person: FamilyData = await self.fetch_individual_family_data(url)
    #     tasks = [
    #         self.fetch_individual_family_data(person.url)
    #         for person in first_person.family_members
    #     ]
    #     return await asyncio.gather(*tasks)
    #
    # def fetch_html_sync(self, url: str) -> str:
    #     """
    #     Asynchronously fetches HTML content from a given URL.
    #
    #     Args:
    #     url (str): The URL of the webpage to be processed.
    #
    #     Returns:
    #     str: HTML content of the webpage.
    #     """
    #
    #     # Adjust the maximum wait time (in seconds) as necessary
    #     max_wait_time = 30
    #
    #     service = Service()
    #     options = webdriver.ChromeOptions()
    #     options.add_argument("--headless")
    #     options.add_argument("--no-sandbox")
    #     options.add_argument("--disable-dev-shm-usage")
    #     driver = webdriver.Chrome(service=service, options=options)
    #     WebDriverWait(driver, max_wait_time).until(
    #         EC.visibility_of_element_located((By.ID, "app-content-scroller"))
    #     )
    #
    #     try:
    #         driver.get(url)
    #         return driver.page_source
    #     finally:
    #         driver.quit()
    #
    # async def fetch_html(self, url: str) -> str:
    #     """
    #     Asynchronously fetches HTML content from a given URL.
    #
    #     Args:
    #     url (str): The URL of the webpage to be processed.
    #
    #     Returns:
    #     str: HTML content of the webpage.
    #     """
    #
    #     async with aiohttp.ClientSession() as session:
    #         try:
    #             async with session.get(url, allow_redirects=True) as response:
    #                 response.raise_for_status()  # Raises an error for 4XX/5XX responses
    #                 return await response.text()
    #         except aiohttp.ClientError as e:
    #             print(f"Request failed: {e}")
    #             logging.error(f"Request failed: {e}")
    #             return ""  # or handle the error as needed
    #
    # def extract_main_person_from_html(self, html: str) -> Person:
    #     soup = BeautifulSoup(html, "html.parser")
    #     lifespan_element = soup.find(attrs={"data-testid": "lifespan"})
    #     lifespan = (
    #         lifespan_element.get_text(strip=True)
    #         if lifespan_element is not None
    #         else None
    #     )
    #     person_id_element = soup.find(attrs={"data-testid": "pid"})
    #     person_id = (
    #         person_id_element.get_text(strip=True)
    #         if person_id_element is not None
    #         else None
    #     )
    #     url = f"https://ancestors.familysearch.org/en/{person_id}"  # TO DO: Move to Person class.
    #     if person_id is None:
    #         print("Test")
    #     # Need to verify that the first name element that matches here is the main person's name and not a sibling, etc.
    #     full_name = soup.find(attrs={"data-testid": "fullName"}).get_text(strip=True)
    #     new_person = Person(
    #         id=person_id, full_name=full_name, url=url, lifespan=lifespan
    #     )
    #     return new_person
    #
    # def extract_person_data_from_html(self, html: str) -> List[Person]:
    #     """
    #     Extracts unique IDs and full names from anchor elements with a specific attribute from HTML content,
    #     and returns a list of Pydantic objects containing this data.
    #
    #     Args:
    #     html (str): HTML content of the webpage.
    #
    #     Returns:
    #     List[Person]: A list of `Person` objects with `id` and `fullName` attributes.
    #     """
    #     soup = BeautifulSoup(html, "html.parser")
    #     people = []
    #     for a in soup.find_all("a", attrs={"data-testid": "nameLink"}):
    #         href = a.get("href", "")
    #         id_part = href.split("/")[-1]
    #         fullName = (
    #             a.find(attrs={"data-testid": "fullName"}).get_text(strip=True)
    #             if a.find(attrs={"data-testid": "fullName"})
    #             else ""
    #         )
    #         if id_part and fullName:
    #             people.append(
    #                 Person(
    #                     id=id_part,
    #                     full_name=fullName,
    #                     url=f"https://ancestors.familysearch.org/en/{id_part}",
    #                 )
    #             )
    #     return people
    #
    # def extract_life_of_card_text_from_html(self, html: str) -> Optional[str]:
    #     """
    #     Extracts and concatenates the texts of deeply nested inner elements of the element with the
    #     attribute data-testid="life-of-card" from HTML content.
    #
    #     Args:
    #     html (str): HTML content of the webpage.
    #
    #     Returns:
    #     Optional[str]: A concatenated string of texts from nested inner elements, separated by ": ".
    #     """
    #     soup = BeautifulSoup(html, "html.parser")
    #     life_of_card_element = soup.find(attrs={"data-testid": "life-of-card"})
    #     if life_of_card_element:
    #         texts = self.extract_text_from_elements(life_of_card_element)
    #         return ": ".join(filter(None, texts))
    #     else:
    #         return None
    #
    # def extract_text_from_elements(self, element) -> List[str]:
    #     """
    #     Recursively extracts text from nested elements.
    #
    #     Args:
    #     element (bs4.element.Tag): The BeautifulSoup element to extract text from.
    #
    #     Returns:
    #     List[str]: A list of strings containing the text from each nested element.
    #     """
    #     if isinstance(element, NavigableString):
    #         return [element.strip()]
    #     if element.name in ["style", "script"]:  # Ignore script and style elements
    #         return []
    #
    #     texts = []
    #     for child in element.children:
    #         texts.extend(self.extract_text_from_elements(child))
    #     return texts
    #
    # async def extract_family_data(self, url: str) -> FamilyData:
    #     """
    #     Asynchronously extracts life summary and family member information from a given URL and
    #     returns it as a FamilyData Pydantic object.
    #
    #     Args:
    #     url (str): The URL of the webpage to be processed.
    #
    #     Returns:
    #     FamilyData: A Pydantic object containing life summary and a list of family members.
    #     """
    #     if url is not None:
    #         html = await self.fetch_html(url)
    #         if "The specified key does not exist" in html:
    #             print("Unable to retrieve data from website - expected response")
    #         else:
    #             print(f"Got response for {url}")
    #         life_summary = self.extract_life_of_card_text_from_html(html)
    #         main_person = self.extract_main_person_from_html(html)
    #         family_members = self.extract_person_data_from_html(html)
    #
    #         return FamilyData(
    #             life_summary=life_summary,
    #             family_members=family_members,
    #             main_person=main_person,
    #         )
    #     else:
    #         logging.info("Got None for url")

    def create_family_vector_index(
        self,
        row: dict,
        vector_store: AstraDB,
    ):
        try:
            logging.debug(f"Creating vector index for row: {row}")
            metadata = {
                "date": row["date"],
                "place": row["place"],
                "recordType": row["recordType"],
                "languages": row["languages"],
            }
            parsed_text = row["parsed_text"]
            my_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=250
            )
            wiki_document = Document(page_content=parsed_text, metadata=metadata)
            wiki_docs = my_splitter.transform_documents([wiki_document])
            vector_store.add_documents(wiki_docs)
            logging.info("Document added to vector store successfully.")
        except Exception as e:
            logging.error(f"Error in creating vector index: {e}", exc_info=True)

    async def create_family_vector_index_async(
        self,
        row: dict,
        vector_store: AstraDB,
    ):
        try:
            logging.debug(f"Creating vector index for row: {row}")
            metadata = {
                "date": row["date"],
                "place": row["place"],
                "recordType": row["recordType"],
                "languages": row["languages"],
            }
            parsed_text = row["parsed_text"]
            my_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=250
            )
            wiki_document = Document(page_content=parsed_text, metadata=metadata)
            wiki_docs = my_splitter.transform_documents([wiki_document])
            await vector_store.add_documents(wiki_docs)
            logging.info("Document added to vector store successfully.")
        except Exception as e:
            logging.error(f"Error in creating vector index: {e}", exc_info=True)

    async def create_family_vector_index_async_v2(
        self,
        row: dict,
        vector_store: AstraDB,
    ):
        try:
            logging.debug(f"Creating vector index for row: {row}")

            parsed_text = row["parsed_text"]
            doc = self.nlp(parsed_text)
            # Extract names
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            csv_name_string = ", ".join(([s.replace(" ┃", "") for s in names]))
            metadata = {
                "date": row["date"],
                "place": row["place"],
                "recordType": row["recordType"],
                "languages": row["languages"],
                "names": csv_name_string,
                "content": parsed_text,
            }
            my_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=250
            )
            wiki_document = Document(page_content=csv_name_string, metadata=metadata)
            wiki_docs = my_splitter.transform_documents([wiki_document])
            await vector_store.add_documents(wiki_docs)
            logging.info("Document added to vector store successfully.")
        except Exception as e:
            logging.error(f"Error in creating vector index: {e}", exc_info=True)

    async def create_family_vector_index_async_v3(
        self,
        row: dict,
        vector_store: AstraDB,
    ):
        try:
            logging.debug(f"Creating vector index for row: {row}")

            parsed_text = row["parsed_text"]
            # doc = self.nlp(parsed_text)
            # Extract names
            # names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            # csv_name_string = ", ".join(
            #     ([s.replace("┃", "").replace("  ", " ") for s in names])
            # )
            cleaned_names = [name.replace("┃", "").strip() for name in row["names"]]
            clean_parsed_text = (
                parsed_text.replace("┃", "")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace(" .", ".")
                .replace(" ,", ".")
                .strip()
            )
            metadata = {
                "date": row["date"],
                "place": row["place"],
                "min_yyyy": int(row["min_yyyy"])
                if row["min_yyyy"] is not None
                else None,
                "max_yyyy": int(row["max_yyyy"])
                if row["max_yyyy"] is not None
                else None,
                "recordType": row["recordType"],
                "languages": row["languages"],
                "names": cleaned_names,
            }
            my_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=0
            )
            wiki_document = Document(page_content=clean_parsed_text, metadata=metadata)
            wiki_docs = my_splitter.transform_documents([wiki_document])
            await vector_store.add_documents(wiki_docs)
            logging.info("Document added to vector store successfully.")
        except Exception as e:
            logging.error(f"Error in creating vector index: {e}", exc_info=True)

    def process_json_files(
        self, directory: str, vector_store: AstraDB, batch_size: int = 20
    ) -> int:
        total_count = 0

        try:
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    file_path = os.path.join(directory, filename)
                    batch = []
                    file_count = 0

                    logging.info(f"Starting to process file: {filename}")

                    with open(file_path, "r") as file:
                        for line in file:
                            try:
                                json_obj = json.loads(line)
                                if json_obj is not None:
                                    batch.append(json_obj)
                                    file_count += 1
                                    logging.debug(
                                        f"Added JSON object to batch: {json_obj}"
                                    )

                                    if len(batch) == batch_size:
                                        logging.debug(
                                            "Batch size reached, starting to process batch"
                                        )
                                        for row in batch:
                                            self.create_family_vector_index(
                                                row, vector_store
                                            )
                                        total_count += len(batch)
                                        logging.debug(
                                            f"Batch processed, total count: {total_count}"
                                        )
                                        batch.clear()
                            except json.JSONDecodeError as ex:
                                logging.error(
                                    f"Invalid JSON format in line from file {filename}: {ex}",
                                    exc_info=True,
                                )

                    if batch:
                        logging.debug("Processing remaining batch")
                        for row in batch:
                            self.create_family_vector_index(row, vector_store)
                        total_count += len(batch)
                        logging.debug(
                            f"Remaining batch processed, total count: {total_count}"
                        )

                    logging.info(
                        f"Completed processing {filename}. Records processed from this file: {file_count}"
                    )
                    logging.info(f"Total records processed so far: {total_count}")
        except Exception as e:
            logging.error(f"Error in process_json_files: {e}", exc_info=True)

        logging.info(f"Processing complete. Total records processed: {total_count}")
        return total_count

    async def process_json_files_async(
        self, directory: str, vector_store: AstraDB, batch_size: int = 5
    ) -> int:
        total_count = 0
        # Load the English language model
        try:
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    file_path = os.path.join(directory, filename)
                    batch = []
                    file_count = 0

                    logging.info(f"Starting to process file: {filename}")

                    with open(file_path, "r") as file:
                        for line in file:
                            try:
                                json_obj = json.loads(line)
                                if json_obj is not None:
                                    batch.append(json_obj)
                                    file_count += 1
                                    logging.debug(
                                        f"Added JSON object to batch: {json_obj}"
                                    )

                                    if len(batch) == batch_size:
                                        logging.debug(
                                            "Batch size reached, starting to process batch"
                                        )
                                        await asyncio.gather(
                                            *[
                                                self.create_family_vector_index_async_v3(
                                                    row, vector_store
                                                )
                                                for row in batch
                                            ]
                                        )
                                        total_count += len(batch)
                                        logging.debug(
                                            f"Batch processed, total count: {total_count}"
                                        )
                                        batch.clear()
                            except json.JSONDecodeError as ex:
                                logging.error(
                                    f"Invalid JSON format in line from file {filename}: {ex}",
                                    exc_info=True,
                                )

                    if batch:
                        logging.debug("Processing remaining batch")
                        await asyncio.gather(
                            *[
                                self.create_family_vector_index_async_v2(
                                    row, vector_store
                                )
                                for row in batch
                            ]
                        )
                        total_count += len(batch)
                        logging.debug(
                            f"Remaining batch processed, total count: {total_count}"
                        )

                    logging.info(
                        f"Completed processing {filename}. Records processed from this file: {file_count}"
                    )
                    logging.info(f"Total records processed so far: {total_count}")
        except Exception as e:
            logging.error(f"Error in process_json_files: {e}", exc_info=True)

        logging.info(f"Processing complete. Total records processed: {total_count}")
        return total_count


def extract_text_from_elements(element) -> List[str]:
    """
    Recursively extracts text from nested elements.

    Args:
    element (bs4.element.Tag): The BeautifulSoup element to extract text from.

    Returns:
    List[str]: A list of strings containing the text from each nested element.
    """
    if isinstance(element, NavigableString):
        return [element.strip()]
    if element.name in ["style", "script"]:  # Ignore script and style elements
        return []

    texts = []
    for child in element.children:
        texts.extend(extract_text_from_elements(child))
    return texts


def extract_life_of_card_text(url: str) -> Optional[str]:
    """
    Extracts and concatenates the texts of deeply nested inner elements of the element with the
    attribute data-testid="life-of-card", separated by ": " from a given URL.

    This function sends a GET request to the specified URL, parses the HTML content, and finds the first element
    with the attribute `data-testid="life-of-card"`. It then recursively extracts texts from each deeply nested
    inner element and concatenates them, separated by ": ".

    Args:
    url (str): The URL of the webpage to be processed.

    Returns:
    Optional[str]: A concatenated string of texts from nested inner elements, separated by ": ", or None if the
    request fails or no such element is found.
    """
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        life_of_card_element = soup.find(attrs={"data-testid": "life-of-card"})
        if life_of_card_element:
            texts = extract_text_from_elements(life_of_card_element)
            return ": ".join(filter(None, texts))
        else:
            return None
    else:
        return None


class Person(BaseModel):
    id: str
    full_name: str
    url: str


async def fetch_html(url: str) -> str:
    """
    Asynchronously fetches HTML content from a given URL.

    Args:
    url (str): The URL of the webpage to be processed.

    Returns:
    str: HTML content of the webpage.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


def extract_person_data_from_html(html: str) -> List[Person]:
    """
    Extracts unique IDs and full names from anchor elements with a specific attribute from HTML content,
    and returns a list of Pydantic objects containing this data.

    Args:
    html (str): HTML content of the webpage.

    Returns:
    List[Person]: A list of `Person` objects with `id` and `fullName` attributes.
    """
    soup = BeautifulSoup(html, "html.parser")
    people = []
    for a in soup.find_all("a", attrs={"data-testid": "nameLink"}):
        href = a.get("href", "")
        id_part = href.split("/")[-1]
        fullName = (
            a.find(attrs={"data-testid": "fullName"}).get_text(strip=True)
            if a.find(attrs={"data-testid": "fullName"})
            else ""
        )
        if id_part and fullName:
            people.append(
                Person(
                    id=id_part,
                    full_name=fullName,
                    url=f"https://ancestors.familysearch.org/en/{id_part}",
                )
            )
    return people


def extract_life_of_card_text_from_html(html: str) -> Optional[str]:
    """
    Extracts and concatenates the texts of deeply nested inner elements of the element with the
    attribute data-testid="life-of-card" from HTML content.

    Args:
    html (str): HTML content of the webpage.

    Returns:
    Optional[str]: A concatenated string of texts from nested inner elements, separated by ": ".
    """
    soup = BeautifulSoup(html, "html.parser")
    life_of_card_element = soup.find(attrs={"data-testid": "life-of-card"})
    if life_of_card_element:
        texts = extract_text_from_elements(life_of_card_element)
        return ": ".join(filter(None, texts))
    else:
        return None


# Assuming the Person class is already defined as earlier
class FamilyData(BaseModel):
    life_summary: Optional[str]
    family_members: List[Person]


async def extract_family_data(url: str) -> FamilyData:
    """
    Asynchronously extracts life summary and family member information from a given URL and
    returns it as a FamilyData Pydantic object.

    Args:
    url (str): The URL of the webpage to be processed.

    Returns:
    FamilyData: A Pydantic object containing life summary and a list of family members.
    """
    html = await fetch_html(url)
    life_summary = extract_life_of_card_text_from_html(html)
    family_members = extract_person_data_from_html(html)

    return FamilyData(life_summary=life_summary, family_members=family_members)


async def fetch_individual_family_data(url: str) -> FamilyData:
    """
    Asynchronously fetches family data for an individual's URL.

    Args:
    url (str): URL of the individual's family data page.

    Returns:
    FamilyData: Extracted family data for the individual.
    """
    return await extract_family_data(url)


async def extract_extended_family_data(url: str) -> List[FamilyData]:
    """
    Asynchronously extracts extended family data for each family member in the given FamilyData object.

    Args:
    family_data (FamilyData): The initial family data containing family members' URLs.

    Returns:
    List[FamilyData]: A list of FamilyData objects for each family member's extended family.
    """
    first_person: FamilyData = await fetch_individual_family_data(url)
    tasks = [
        fetch_individual_family_data(person.url)
        for person in first_person.family_members
    ]
    return await asyncio.gather(*tasks)


async def extract_deep_extended_family_data(url: str) -> List[FamilyData]:
    """
    Recursively extracts extended family data for each family member, going two levels deep.

    Args:
    url (str): The initial URL to start the extraction from.

    Returns:
    List[FamilyData]: A list of FamilyData objects for each family member's extended family,
                      including the extended family of each member.
    """
    # First level of family data extraction
    first_level_family_data = await fetch_individual_family_data(url)
    second_level_data = []

    # Second level of family data extraction for each family member
    for person in first_level_family_data.family_members:
        family_next_data: List[FamilyData] = await extract_extended_family_data(
            person.url
        )
        second_level_data.append(family_next_data)

    # Combine first level and second level results
    all_family_data = [first_level_family_data] + second_level_data
    return all_family_data


def consolidate_life_summaries(
    family_data_list: Union[List[FamilyData], List[List[FamilyData]]],
    seen_summaries: Set[str] = None,
) -> str:
    """
    Consolidates all unique life_summary values from a list of FamilyData objects into a single string,
    with each summary separated by a new line character. Handles nested lists and eliminates duplicates.

    Args:
    family_data_list (Union[List[FamilyData], List[List[FamilyData]]]): A list of FamilyData objects or nested lists of FamilyData objects.
    seen_summaries (Set[str]): A set to keep track of already seen life summaries.

    Returns:
    str: A string containing all unique life_summaries, separated by new lines.
    """
    if seen_summaries is None:
        seen_summaries = set()

    summaries = []

    for family in family_data_list:
        if (
            isinstance(family, FamilyData)
            and family.life_summary
            and family.life_summary not in seen_summaries
        ):
            summaries.append(family.life_summary)
            seen_summaries.add(family.life_summary)
        elif isinstance(family, list):
            nested_summaries = consolidate_life_summaries(family, seen_summaries)
            if nested_summaries:
                summaries.append(nested_summaries)

    return "\n".join(summaries)
