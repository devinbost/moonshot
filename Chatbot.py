from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from DataAccess import DataAccess
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)


class Chatbot:
    # Need to refactor this class at some point to support multiple LLM providers
    def __init__(self, data_access: DataAccess):
        self.data_access = data_access
        self.model_id = "google/flan-t5-xxl"

        self.credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": os.getenv("IBM_API_SECRET"),
        }

        gen_parms = {
            "DECODING_METHOD": "greedy",
            "MIN_NEW_TOKENS": 1,
            "MAX_NEW_TOKENS": 50,
        }

        # I occasionally get a 443 CA error that appears to be intermittent. Need exponential backoff/retry.
        self.legacy_model = Model(
            self.model_id, self.credentials, gen_parms, os.getenv("IBM_PROJECT_ID")
        )
        self.langchain_model = WatsonxLLM(model=self.legacy_model)

        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.langchain_model,
            chain_type="stuff",
            retriever=self.data_access.vector_store.as_retriever(),
            return_source_documents=True,
        )

    def similarity_search_top_k(self, question: str, top_k: int):
        results = self.data_access.vector_store.similarity_search(question, top_k)
        return results

    def print_results(self, results):
        for row in results:
            print(f"""{row.page_content}\n""")

    def runInference(self, question: str):
        output = self.rag_chain({"query": question})
        print(output)
        return output
