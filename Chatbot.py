from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from DataAccess import DataAccess
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


class Chatbot:
    # Need to refactor this class at some point to support multiple LLM providers
    def __init__(self, data_access: DataAccess):
        self.memory = ConversationBufferMemory(
            return_messages=True
        )  # Set to false if we want to return a string instead of a list
        provider = os.getenv("PROVIDER")
        self.data_access = data_access
        if provider == "IBM":
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
        elif provider == "OPENAI":
            self.langchain_model = ChatOpenAI(temperature=0, model_name="gpt-4")

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

    def runInference(self, question: str) -> dict:
        chat_history = ConversationBufferMemory(memory_key="chat_history")
        template = f"""You are a nice chatbot having a conversation with a human.

        Previous conversation:
        {chat_history}

        New human question: {question}
        Response:"""
        self.memory.chat_memory.add_user_message(question)
        bot_response = self.rag_chain({"query": question})
        # Should I run vector search on the new query and then combine results with the prior?
        # Or, should I run vector search on the combined conversation?
        print(bot_response)
        # We may want to also capture bot_response["source_documents"] for analytics later
        self.memory.chat_memory.add_ai_message(bot_response["result"])
        return bot_response
