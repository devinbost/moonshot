import json
import logging
from operator import itemgetter
from typing import List, Dict

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

import PromptFactory
from ChainFactory import ChainFactory
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
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableLambda

from pydantic_models.PropertyInfo import PropertyInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


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

    def answer_customer(self):
        fake_data_access: DataAccess = DataAccess()
        model: ChatOpenAI = ChatOpenAI(model_name="gpt-4-1106-preview")
        # Faking the user_info for now. Will pull from UI & DB later.
        user_info: UserInfo = UserInfo(
            properties=[
                PropertyInfo(
                    property_name="age", property_type="int", property_value=30
                ),
                PropertyInfo(
                    property_name="name",
                    property_type="text",
                    property_value="John Smith",
                ),
                PropertyInfo(
                    property_name="phone_number",
                    property_type="text",
                    property_value="555-555-5555",
                ),
                PropertyInfo(
                    property_name="email",
                    property_type="text",
                    property_value="johndoe@example.com",
                ),
                PropertyInfo(
                    property_name="address",
                    property_type="text",
                    property_value="123 Main St, Anytown, USA",
                ),
                PropertyInfo(
                    property_name="account_status",
                    property_type="text",
                    property_value="Active",
                ),
                PropertyInfo(
                    property_name="plan_type",
                    property_type="text",
                    property_value="Unlimited Data Plan",
                ),
            ]
        )
        relevant_table_chain: Runnable = (
            {
                "TableList": RunnableLambda(
                    fake_data_access.get_table_schemas_in_db_v2
                ),
                "UserInfo": itemgetter("user_info_not_summary"),
            }
            | PromptFactory.build_table_identification_prompt()
            | model
            | StrOutputParser()
            | RunnableLambda(PromptFactory.clean_string_v2)
            | RunnableLambda(fake_data_access.map_tables_and_populate)
        )
        relevant_tables: List[TableSchema] = relevant_table_chain.invoke(
            {"user_info_not_summary": user_info}
        )
        factory: ChainFactory = ChainFactory()
        all_user_table_summaries: List[str] = []

        all_collection_keywords: Dict = fake_data_access.get_path_segment_keywords()
        all_table_insights: List[str] = []
        for table in relevant_tables:
            print(f"Found relevant table: {table}")
            logging.info(f"Found relevant table: {table}")
            table_summarization_chain: Runnable = factory.build_summarization_chain(
                model, fake_data_access, table
            )
            table_summarization: str = table_summarization_chain.invoke(
                {"user_info_not_summary": user_info}
            )
            print(f"Here is the table summary: {table_summarization}")
            logging.info(f"Here is the table summary: {table_summarization}")
            all_user_table_summaries.append(table_summarization)

            predicate_identification_chain: Runnable = (
                factory.build_collection_predicate_chain_non_parallel_v2(
                    model, table_summarization, all_collection_keywords
                )
            )
            collection_predicates: str = predicate_identification_chain.invoke({})
            topic_summaries_for_table: List[str] = []
            print(f"Collection predicates are: {collection_predicates}")
            logging.info(f"Collection predicates are: {collection_predicates}")
            for predicate in collection_predicates:
                print(f"Topic is: {predicate}")
                logging.info(f"Topic is: {predicate}")
                search_results_for_topic: str = fake_data_access.filtered_ANN_search(
                    predicate, table_summarization
                )
                print(
                    f"Here were search results for that topic: {search_results_for_topic}"
                )
                logging.info(
                    f"Here were search results for that topic: {search_results_for_topic}"
                )
                summarization_of_topic_chain: Runnable = (
                    factory.build_vector_search_summarization_chain(
                        model, search_results_for_topic
                    )
                )
                summarization_of_topic: str = summarization_of_topic_chain.invoke({})
                print(f"Here is the summary for the topic: {summarization_of_topic}")
                logging.info(
                    f"Here is the summary for the topic: {summarization_of_topic}"
                )
                topic_summaries_for_table.append(summarization_of_topic)
            topic_summaries_for_table_as_string: str = json.dumps(
                topic_summaries_for_table
            )
            summarization_of_findings_for_table: Runnable = (
                factory.build_vector_search_summarization_chain(
                    model, topic_summaries_for_table_as_string
                )
            )
            insights_on_table: str = summarization_of_findings_for_table.invoke({})
            print(
                f"Here is the full summarization for the table across its topics: {insights_on_table}"
            )
            logging.info(
                f"Here is the full summarization for the table across its topics: {insights_on_table}"
            )
            all_table_insights.append(insights_on_table)

        recommendation_chain: Runnable = (
            factory.build_final_recommendation_chain_non_parallel(model)
        )
        recommendation: str = recommendation_chain.invoke(
            {
                "user_summary": all_user_table_summaries,
                "business_summary": all_table_insights,
                "user_messages": ["Hi, I'm having trouble with my phone network"],
            }
        )

        print(f"Recommendation to the user is: {recommendation}")
        logging.info(f"Recommendation to the user is: {recommendation}")
