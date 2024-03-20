import asyncio
import logging
from typing import List, Dict, Any
import hashlib
import json
from astrapy.db import AstraDBCollection, AsyncAstraDBCollection, AsyncAstraDB, AstraDB
import numpy as np
from core.adapters.EmbeddingInterface import EmbeddingInterface


class CollectionManager:
    """
    This class is primarily used for manual interaction with collections, like when saving and
    retrieving prompts from an AstraDB collection via vector search.
    """

    def __init__(
        self, astrapy_db: AstraDB | AsyncAstraDB, embedding_interface: EmbeddingInterface, collection_name: str
    ):
        self.astrapy_db = astrapy_db
        self.embedding = embedding_interface
        self.collection_name = collection_name

    def save_prompt(self, prompt: str) -> None:
        """
        Save a given prompt to the AstraDB collection 'prompts', encoding the prompt using embeddings and generating a unique identifier.
        Parameters:
            prompt (str): The prompt to be saved in the database.
        """
        mycollections = self.astrapy_db.get_collections()["status"]["collections"]
        if "prompts" not in mycollections:
            collection = self.astrapy_db.create_collection(
                collection_name="prompts", dimension=384
            )
        else:
            collection = AstraDBCollection(
                collection_name="prompts", astra_db=self.astrapy_db
            )
        # Workaround due to strange uuid bug:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        vector = self.embedding.embed_text(prompt).tolist()
        collection.insert_one({"_id": prompt_hash, "prompt": prompt, "$vector": vector})

    def get_matching_prompts(self, match: str) -> List[Dict[str, Any]]:
        """
        Retrieve and return matching prompts from the AstraDB collection 'prompts' based on the similarity to the given match string.
        Parameters:
            match (str): The string to match against the stored prompts.
        Returns:
            List[Dict[str, Any]]: A list of prompts from the database that closely match the given string.
        """
        mycollections = self.astrapy_db.get_collections()["status"]["collections"]
        if "prompts" not in mycollections:
            collection = self.astrapy_db.create_collection(
                collection_name="prompts", dimension=384
            )
        else:
            collection = AstraDBCollection(
                collection_name="prompts", astra_db=self.astrapy_db
            )
        vector = self.embedding.embed_text(match).tolist()
        results = collection.vector_find(vector, limit=10)
        return results
        # Query DB for prompts
    async def filtered_ANN_search_async(
        self, collection_filter: Dict[str, str], user_summary: Any, limit: int
    ) -> str:
        """
        Perform an Approximate Nearest Neighbor (ANN) search with a filter and user summary asynchronously,
        returning the results as a JSON string, using asyncio.to_thread to run in a separate thread.
        Parameters:
            collection_filter (Dict[str, str]): A dictionary to filter the collection.
            user_summary (Any): A summary provided by the user, used in the search query.
        Returns:
            str: A JSON string representing the search results.
        """
        user_summary_string = json.dumps(user_summary)
        input_vector: List[float] = self.embedding.embed_text(
            user_summary_string
        ).tolist()
        collection = AsyncAstraDBCollection(
            collection_name="sitemapls", astra_db=self.astrapy_db
        )

        try:
            results: List[Dict[str, Any]] = await collection.vector_find(
                vector=input_vector,
                filter=collection_filter,
                limit=limit,
            )
            for result in results:
                print(
                    "nlp_keywords are: " + result["metadata"]["nlp_keywords"]
                )  # TODO: Remove ^ after testing
                print("content is: " + result["content"])
            result_contents = [result["content"] for result in results]
            return json.dumps(result_contents)
        except Exception as ex:
            logging.error("Error reading from DB. Exception: " + str(ex))
            return ""

    async def ANN_search_async(
        self, question: str, limit: int
    ) -> str:
        """
        Perform an Approximate Nearest Neighbor (ANN) search with a filter and user summary asynchronously,
        returning the results as a JSON string, using asyncio.to_thread to run in a separate thread.
        Parameters:
            collection_filter (Dict[str, str]): A dictionary to filter the collection.
            question (str): The search query.
        Returns:
            str: A JSON string representing the search results.
        """
        input_vector: List[float] = self.embedding.embed_text(
            question
        )
        if isinstance(input_vector, np.ndarray):
            input_vector = input_vector.tolist()

        collection = AsyncAstraDBCollection(
            collection_name=self.collection_name, astra_db=self.astrapy_db
        )

        try:
            results: List[Dict[str, Any]] = await collection.vector_find(
                vector=input_vector,
                limit=limit,
            )
            result_contents = [result["content"] for result in results]
            return json.dumps(result_contents)
        except Exception as ex:
            logging.error("Error reading from DB. Exception: " + str(ex))
            return ""