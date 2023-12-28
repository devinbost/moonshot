import logging

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

from DataAccess import DataAccess
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import AstraDB
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class Chatbot:
    # Need to refactor this class at some point to support multiple LLM providers
    def __init__(self, data_access: DataAccess):
        print("ran __init__ on Chatbot")
        self.chat_history = []
        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history", output_key="answer"
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

        self.rag_chain = ConversationalRetrievalChain.from_llm(
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

    def run_inference_astrapy(
        self,
        terms_for_ann: str,
        ann_length: int,
        collection: str,
        question: str,
    ) -> object:
        vector_store = self.data_access.setupVectorStoreNew(collection=collection)
        results = vector_store.similarity_search(terms_for_ann, k=ann_length)
        for doc in results:
            doc.page_content = doc.page_content.replace("{", "{{").replace("}", "}}")
        concatenated_content = "\nEXAMPLE: \n".join(
            [doc.page_content for doc in results]
        )

        template = f"""{{question}}
        
        EXAMPLES: 
        {concatenated_content}
        """
        # Create a prompt from the template
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize the ChatOpenAI model
        model = ChatOpenAI()
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"question": question})  # Replace with your question
        logging.info(response)
        return response

    def runInference(self, question: str) -> dict:
        topK = self.similarity_search_top_k(question, 10)
        bot_response = self.rag_chain(
            {
                "question": "Answer as if you're an expert in the resources provided as context below. Also, if you're not sure, just answer based on what you know from the information below.-- "
                + question,
                "chat_history": self.chat_history,
            }
        )
        self.chat_history.append((question, bot_response["answer"]))
        # Should I run vector search on the new query and then combine results with the prior?
        # Or, should I run vector search on the combined conversation?
        print(bot_response)
        # We may want to also capture bot_response["source_documents"] for analytics later
        return bot_response
