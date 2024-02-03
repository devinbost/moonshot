from astrapy.db import AstraDBCollection

import unittest

from sentence_transformers import SentenceTransformer


class TestAstrapy(unittest.TestCase):
    def test_astrapy_execution(self):
        context = {}
        code = """
import os
from astrapy.db import AstraDB as AstraPyDB
db = AstraPyDB(token=os.getenv("ASTRA_TOKEN"), api_endpoint=os.getenv("ASTRA_ENDPOINT"))
"""
        exec(code, context)
        db = context["db"]
        print(db)

    def test_astrapy_find(self):
        import os
        from astrapy.db import AstraDB as AstraPyDB

        db = AstraPyDB(
            token=os.getenv("ASTRA_TOKEN"),
            api_endpoint=os.getenv("ASTRA_ENDPOINT"),
        )
        embedding_model = "all-MiniLM-L12-v2"
        embedding_direct = SentenceTransformer(
            "sentence-transformers/" + embedding_model
        )
        example_msg = (
            "Hi, I'm having an issue with my iPhone 6. The network isn't working"
        )
        input_vector = embedding_direct.encode(example_msg).tolist()

        # mycollections = db.get_collections()["status"]["collections"]

        # Assume that all collections are relevant for now.
        # Later, we will use a chain to get only the relevant ones.
        collection = AstraDBCollection(collection_name="sitemapls", astra_db=db)
        results = collection.vector_find(
            vector=input_vector,
            filter={"metadata.path_segment_1": "support"},
            limit=100,
        )
        print(results)
