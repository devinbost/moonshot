import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Any, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

import PromptFactory
from ChainFactory import ChainFactory
from DataAccess import DataAccess
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
        """
        Initialize the Chatbot with a data access object.
        Parameters:
            data_access (DataAccess): The data access object to interact with the database.
        """
        self.our_responses = [""]
        self.column = None
        self.user_messages = [""]
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
        self.relevant_table_cache = None

    def similarity_search_top_k(
        self, question: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Conducts a similarity search for the given question and returns the top K results.
        Parameters:
            question (str): The question to search for.
            top_k (int): The number of top results to return.
        Returns:
            List[Dict[str, Any]]: The top K similar results from the similarity search.
        """
        results = self.data_access.vector_store.similarity_search(question, top_k)
        return results

    def print_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Prints the search results.
        Parameters:
            results (List[Dict[str, Any]]): The search results to be printed.
        """
        for row in results:
            print(f"""{row.page_content}\n""")

    def run_inference_astrapy(
        self,
        terms_for_ann: str,
        ann_length: int,
        collection: str,
        question: str,
    ) -> Any:
        """
        Runs inference using the AstraPy model with Approximate Nearest Neighbor (ANN) search.
        Parameters:
            terms_for_ann (str): Terms to be used for ANN search.
            ann_length (int): The number of results to retrieve from the ANN search.
            collection (str): The collection to search within.
            question (str): The question to be answered.
        Returns:
            Any: The response from the inference.
        """
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

    def runInference(self, question: str) -> Dict[str, Any]:
        """
        Runs inference to answer a given question using the model and data access object.
        Parameters:
            question (str): The question to be answered.
        Returns:
            Dict[str, Any]: The response from the inference.
        """
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

    def log_response(
        self,
        entry_type: str,
        bot_message: str | List[dict[str, str]] | dict[str, List[str]],
        show_time=False,
    ) -> None:
        """
        Logs the bot's response.
        Parameters:
            bot_message (str): The message to be logged.
        """
        topics = ""
        print(bot_message)
        logging.info(bot_message)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f")
        if show_time:
            if entry_type == "Relevant Topics":
                for topic in bot_message:
                    topic_value = next(iter(topic.values()))
                    topics += topic_value + ", "
                self.column.text_area(
                    entry_type,
                    value=f"Company docs were found on these relevant topics:\n\n {topics}. Time: {timestamp}",
                    key=datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f"),
                )
            else:
                self.column.text_area(
                    entry_type,
                    value=bot_message + f". Time: {timestamp}",
                    key=datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f"),
                )
        else:
            if entry_type == "Predicates":
                for topic in bot_message:
                    topic_value = next(iter(topic.values()))
                    topics += topic_value + ", "
                self.column.text_area(
                    entry_type,
                    value=f"Company docs found on these relevant topics:\n\n {topics}",
                    key=datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f"),
                )
            if entry_type == "Relevant Topics List":
                result = ", ".join(
                    item for sublist in bot_message.values() for item in sublist
                )
                self.column.text_area(
                    entry_type,
                    value=f"These were the most relevant topics:\n\n {result}.",
                    key=datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f"),
                )
            if entry_type == "Relevant predicates List":
                result = ", ".join(value for d in bot_message for value in d.values())
                self.column.text_area(
                    entry_type,
                    value=f"These appeared to be the most relevant of those topics:\n\n {result}.",
                    key=datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f"),
                )
            else:
                self.column.text_area(
                    entry_type,
                    value=bot_message,
                    key=datetime.utcnow().strftime("%Y%m%d-%H-%M-%S-%f"),
                )

    def slice_into_chunks(
        self, collection_group: str, lst, n
    ) -> List[Tuple[str, List[str]]]:
        # Calculate the size of each chunk
        chunk_size = len(lst) // n
        # Handle the case where the list size isn't perfectly divisible by n
        if len(lst) % n != 0:
            chunk_size += 1
            # Create a list of dictionaries
        return [
            (collection_group, lst[i : i + chunk_size])
            for i in range(0, len(lst), chunk_size)
        ]

    def answer_customer(
        self, user_message: str, user_info: UserInfo, column: Any
    ) -> str:
        """
        Provides an answer to the customer's query.
        Parameters:
            user_message (str): The customer's message.
            user_info (UserInfo): The user information.
            column (Any): The UI column to display the response.
        Returns:
            str: The bot's response to the customer.
        """
        self.column = column
        data_access: DataAccess = DataAccess()
        model: ChatOpenAI = ChatOpenAI(model_name="gpt-4-1106-preview")
        model35: ChatOpenAI = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
        self.log_response("Start", "Inspecting hundreds of tables in the database")
        if self.relevant_table_cache is None:
            relevant_table_chain: Runnable = (
                {
                    "TableList": RunnableLambda(data_access.get_table_schemas_in_db_v2),
                    "UserInfo": itemgetter("user_info_not_summary"),
                }
                | PromptFactory.build_table_identification_prompt()
                | model
                | StrOutputParser()
                | RunnableLambda(PromptFactory.clean_string_v2)
                | RunnableLambda(data_access.map_tables_and_populate)
            )
            relevant_tables: List[TableSchema] = relevant_table_chain.invoke(
                {"user_info_not_summary": user_info}
            )
            self.relevant_table_cache = relevant_tables
        factory: ChainFactory = ChainFactory()
        all_user_table_summaries: List[str] = []

        # self.log_response("Status", f"Getting path segment keywords")
        all_collection_keywords: Dict[
            str, List[str]
        ] = data_access.get_path_segment_keywords()
        keyword_list_slices: List[Tuple[str, List[str]]] = []
        # ^^ [('metadata.path_segment_2': ['aqa-case-with-glitter-for-iphone-12-iphone-12-pro', 'business',. . .

        keyword_list_slices: List[Tuple[str, List[str]]] = self.build_keyword_slices(
            all_collection_keywords, keyword_list_slices
        )
        # keyword_list_slices = [
        #     (k, sorted(v, key=lambda x: (x is None, x))) for k, v in keyword_list_slices
        # ]
        keyword_list_slices = [(k, sorted(v)) for k, v in keyword_list_slices]

        all_table_insights: List[str] = []
        for table in self.relevant_table_cache:
            self.log_response(
                "Found Table", f"Found relevant table: {table.table_name}"
            )
            table_summarization_chain: Runnable = factory.build_summarization_chain(
                model, data_access, table
            )
            table_summarization: str = table_summarization_chain.invoke(
                {"user_info_not_summary": user_info}
            )
            self.log_response(
                "Table Summary",
                f"Summary of what we know about customer from this table:\n\n {table_summarization}",
            )
            all_user_table_summaries.append(table_summarization)

            # I need to reduce the number of keywords that get used in the next step.

            # Do this in parallel:
            self.log_response(
                "Inspecting Knowledge Base",
                f"Evaluating thousands of articles across company knowledge base for insights",
            )
            keyword_reduction_chains: List[Tuple[str, Runnable]] = [
                (
                    keyword_list_slice[0],
                    factory.build_keyword_reduction_prompt_chain(
                        model35, table_summarization, keyword_list_slice[1]
                    ),
                )
                for keyword_list_slice in keyword_list_slices
            ]
            # [('metadata.path_segment_2', {
            #   chain
            # }

            with ThreadPoolExecutor() as executor:
                reduced_keyword_list = list(
                    executor.map(
                        lambda keyed_chain: (keyed_chain[0], keyed_chain[1].invoke({})),
                        keyword_reduction_chains,
                    )
                )
            # [('metadata.path_segment_2', '["international-long-distance-faqs", "nokia-2720-v-flip-update", "5g-home-router-trou . . .
            # Aggregate results based on keys
            reduced_segment_keywords = self.reduce_segments(reduced_keyword_list)
            deduped_segment_keywords = {
                key: list(set(value)) for key, value in reduced_segment_keywords.items()
            }

            real_deduped_keys = self.get_real_keys(
                all_collection_keywords, deduped_segment_keywords
            )
            self.log_response("Relevant Topics List", real_deduped_keys)
            real_predicates = []
            for key in deduped_segment_keywords.keys():
                for value in deduped_segment_keywords[key]:
                    if value is not None:
                        real_predicates.append({key: value})

            predicate_identification_chain: Runnable = (
                factory.build_collection_predicate_chain_non_parallel_v2(
                    model, table_summarization, real_deduped_keys
                )
            )

            collection_predicates: str = predicate_identification_chain.invoke({})
            # Need to replace with real predicates

            self.log_response("Relevant predicates List", collection_predicates)
            topic_summaries_for_table: List[str] = []
            # self.log_response("Relevant Topics", collection_predicates)
            self.log_response(
                "Status", "Reading all articles on these topics in knowledge base"
            )
            with ThreadPoolExecutor() as executor:
                topic_summaries_for_table = list(
                    executor.map(
                        lambda inner_predicate: self.process_predicate(
                            inner_predicate,
                            table_summarization,
                            model,
                            factory,
                            data_access,
                        ),
                        collection_predicates,
                    )
                )
            # self.log_response("Status", "Ran thread pool")
            topic_summaries_for_table_as_string = json.dumps(topic_summaries_for_table)
            logging.info(topic_summaries_for_table_as_string)
            print(topic_summaries_for_table_as_string)
            if topic_summaries_for_table_as_string is not None:
                summarization_of_findings_for_table: Runnable = (
                    factory.build_vector_search_summarization_chain(
                        model, topic_summaries_for_table_as_string
                    )
                )
            else:
                print("error")

            insights_on_table: str = summarization_of_findings_for_table.invoke({})
            self.log_response(
                "Summary of Topics",
                f"Here's what we know that might be relevant from reading company docs on all of those topics:\n\n {insights_on_table}",
            )
            all_table_insights.append(insights_on_table)

        recommendation_chain: Runnable = (
            factory.build_final_recommendation_chain_non_parallel(model)
        )
        all_user_table_summaries.append(
            f"User name is: {user_info.to_lcel_json_prefixed()}"
        )

        recommendation: str = recommendation_chain.invoke(
            {
                "user_summary": all_user_table_summaries,
                "business_summary": all_table_insights,
                "user_messages": [user_message],
                "our_responses": self.our_responses,
            }
        )

        self.our_responses.append(recommendation)

        self.log_response(
            "Recommendation", f"Recommendation to the user is: {recommendation}"
        )
        return recommendation

    def get_real_keys(self, all_collection_keywords, deduped_segment_keywords):
        real_deduped_keys = {}
        for key in deduped_segment_keywords.keys():
            for value in deduped_segment_keywords[key]:
                if value in all_collection_keywords[key]:
                    if key in real_deduped_keys:
                        real_deduped_keys[key].append(value)
                    else:
                        real_deduped_keys[key] = [value]
        return real_deduped_keys

    def reduce_segments(self, reduced_keyword_list):
        reduced_segment_keywords = {}
        for key, result in reduced_keyword_list:
            if key in reduced_segment_keywords:
                reduced_segment_keywords[key].extend(result)
            else:
                reduced_segment_keywords[key] = result
        return reduced_segment_keywords

    def build_keyword_slices(self, all_collection_keywords, keyword_list_slices):
        for collection_group in all_collection_keywords.keys():
            list_for_seg: List[str] = all_collection_keywords[collection_group]
            list_slices: List[Tuple[str, List[str]]] = self.slice_into_chunks(
                collection_group, list_for_seg, 20
            )  # Sliced into 20 sublists
            keyword_list_slices.extend(list_slices)
        return keyword_list_slices

    def process_predicate(
        self, predicate, table_summarization, model, factory, data_access
    ):
        # self.log_response(f"Topic is: {predicate}")
        search_results_for_topic: str = data_access.filtered_ANN_search(
            predicate, table_summarization
        )
        # If search_results_for_topic is None or empty string, then return None.

        # self.log_response(
        #     f"Here were search results for that topic: {search_results_for_topic}"
        # )
        summarization_of_topic_chain: Runnable | None = (
            factory.build_vector_search_summarization_chain(
                model, search_results_for_topic
            )
        )
        if summarization_of_topic_chain is not None:
            summarization_of_topic: str = summarization_of_topic_chain.invoke({})
            # self.log_response(
            #     f"Here is the summary for the topic: {summarization_of_topic}"
            # )
            return summarization_of_topic
        else:
            return ""
