import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Any, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

from core.PromptFactory import PromptFactory
from core.ChainFactory import ChainFactory
from core.DataAccess import DataAccess
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from core.CollectionManager import CollectionManager
from core.EmbeddingManager import EmbeddingManager

from core.ConfigLoader import ConfigLoader
from core.VectorStoreFactory import VectorStoreFactory
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

        self.relevant_table_cache = None

    def run_inference_astrapy(
        self,
        terms_for_ann: str,
        ann_length: int,
        collection: str,
        question: str,
        vector_store_factory: VectorStoreFactory,
    ) -> Any:
        """
        Runs inference using the AstraPy model with Approximate Nearest Neighbor (ANN) search.
        Parameters:
            terms_for_ann (str): Terms to be used for ANN search.
            ann_length (int): The number of results to retrieve from the ANN search.
            collection (str): The collection to search within.
            question (str): The question to be answered.
            vector_store_factory (VectorStoreFactory): The factory for creating the vector store.
        Returns:
            Any: The response from the inference.
        """

        vector_store = vector_store_factory.create_vector_store(
            "AstraDB", collection_name=collection
        )
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

    async def answer_customer(
        self,
        user_message: str,
        user_info: UserInfo,
        column: Any,
        col1,
        summarization_limit,
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

        if self.relevant_table_cache is None:
            # Moved the instantiation of models and factories outside of the if-block to avoid unnecessary recreations
            model = ChatOpenAI(model_name="gpt-4-1106-preview")
            model35 = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
            factory = ChainFactory()
            data_access = (
                self.data_access
            )  # Assuming self.data_access is already instantiated in __init__

            self.log_response("Start", "Inspecting hundreds of tables in the database")

            relevant_tables = await factory.get_relevant_tables(
                data_access, model, user_info
            )
            self.relevant_table_cache = relevant_tables

        all_user_table_summaries = []

        all_collection_keywords = await data_access.get_path_segment_keywords_async()
        keyword_list_slices = self.build_keyword_slices(all_collection_keywords)

        process_table_futures = [
            self.process_table_async(
                all_collection_keywords,
                all_user_table_summaries,
                data_access,
                factory,
                keyword_list_slices,
                model,
                model35,
                table,
                user_info,
                summarization_limit,
            )
            for table in self.relevant_table_cache
        ]

        all_table_insights = await asyncio.gather(*process_table_futures)

        recommendation = await self.generate_recommendation_async(
            factory, model, all_user_table_summaries, all_table_insights, user_message
        )
        self.our_responses.append(recommendation)
        self.log_response(
            "Recommendation", f"Recommendation to the user is: {recommendation}"
        )
        bot_chat_area = col1.markdown(recommendation)
        return recommendation

    # async def answer_customer_colbert(
    #     self,
    #     user_message: str,
    #     user_info: UserInfo,
    #     column: Any,
    #     col1,
    #     summarization_limit,
    # ) -> str:
    #     """
    #     Provides an answer to the customer's query.
    #     Parameters:
    #         user_message (str): The customer's message.
    #         user_info (UserInfo): The user information.
    #         column (Any): The UI column to display the response.
    #     Returns:
    #         str: The bot's response to the customer.
    #     """
    #
    #     self.column = column
    #
    #     if self.relevant_table_cache is None:
    #         # Moved the instantiation of models and factories outside of the if-block to avoid unnecessary recreations
    #         model = ChatOpenAI(model_name="gpt-4-1106-preview")
    #         model35 = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    #         factory = ChainFactory()
    #         data_access = (
    #             self.data_access
    #         )  # Assuming self.data_access is already instantiated in __init__
    #
    #         self.log_response("Start", "Inspecting hundreds of tables in the database")
    #
    #         relevant_tables = await factory.get_relevant_tables(
    #             data_access, model, user_info
    #         )
    #         self.relevant_table_cache = relevant_tables
    #
    #     process_question_answer_futures = [
    #         self.process_table_async_colbert(
    #             data_access,
    #             factory,
    #             model,
    #             model35,
    #             table,
    #             user_info,
    #             summarization_limit,
    #             user_message,
    #         )
    #         for table in self.relevant_table_cache
    #     ]
    #
    #     all_qa_insights = await asyncio.gather(*process_question_answer_futures)
    #
    #     recommendation = await self.generate_recommendation_async(
    #         factory, model, all_user_table_summaries, all_qa_insights, user_message
    #     )
    #     self.our_responses.append(recommendation)
    #     self.log_response(
    #         "Recommendation", f"Recommendation to the user is: {recommendation}"
    #     )
    #     bot_chat_area = col1.markdown(recommendation)
    #     return recommendation

    async def process_table_async(
        self,
        all_collection_keywords,
        all_user_table_summaries,
        data_access,
        factory,
        keyword_list_slices,
        model,
        model35,
        table,
        user_info,
        summarization_limit: int,
    ):
        self.log_response("Found Table", f"Found relevant table: {table.table_name}")
        table_summarization = await self.summarize_table_async(
            factory, model, table, user_info
        )
        all_user_table_summaries.append(table_summarization)
        self.log_response(
            "Inspecting Knowledge Base",
            "Evaluating thousands of articles across company knowledge base for insights",
        )
        reduced_keyword_list = await self.reduce_keywords_async(
            model35, factory, table_summarization, keyword_list_slices
        )
        deduped_segment_keywords = self.deduplicate_keywords(reduced_keyword_list)
        real_deduped_keys = self.get_real_keys(
            all_collection_keywords, deduped_segment_keywords
        )
        self.log_response("Relevant Topics List", real_deduped_keys)
        collection_predicates = await self.identify_predicates_async(
            factory, model, table_summarization, real_deduped_keys
        )
        self.log_response("Collection Predicates", collection_predicates)
        topic_summaries_for_table = await self.summarize_topics_async(
            collection_predicates,
            table_summarization,
            model,
            factory,
            data_access,
            summarization_limit,
        )
        self.log_response("Topic summaries", topic_summaries_for_table)
        insights_on_table = await self.summarize_findings_async(
            factory, model, topic_summaries_for_table
        )
        return insights_on_table

    # async def process_table_async_colbert(
    #     self,
    #     data_access,
    #     factory,
    #     keyword_list_slices,
    #     model,
    #     model35,
    #     table,
    #     user_info,
    #     summarization_limit: int,
    # ):
    #     self.log_response("Found Table", f"Found relevant table: {table.table_name}")
    #     customer_questions = await self.summarize_table_and_generation_questions_async(
    #         factory, model, model35, table, user_info
    #     )
    #     self.log_response(
    #         "Inspecting Knowledge Base",
    #         "Evaluating thousands of articles across company knowledge base for insights",
    #     )
    #
    #     doc_summary_answers = await self.answer_questions_async(
    #         customer_questions,
    #         model,
    #         factory,
    #         data_access,
    #         summarization_limit,
    #     )
    #     self.log_response("Documentation summaries", doc_summary_answers)
    #     insights_on_table = await self.summarize_findings_async(
    #         factory, model, topic_summaries_for_table
    #     )
    #     return insights_on_table

    def generate_database_records(self, model: ChatOpenAI):
        chain = (
            {"MissionStatement": itemgetter("mission_statement")}
            | PromptFactory.build_company_description_data()
            | model
            | StrOutputParser()
        )
        return chain

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

    def build_keyword_slices(self, all_collection_keywords):
        keyword_list_slices: List[Tuple[str, List[str]]] = []
        for collection_group in all_collection_keywords.keys():
            list_for_seg: List[str] = all_collection_keywords[collection_group]
            list_slices: List[Tuple[str, List[str]]] = self.slice_into_chunks(
                collection_group, list_for_seg, 20
            )  # Sliced into 20 sublists
            keyword_list_slices.extend(list_slices)
        return keyword_list_slices

    @DeprecationWarning
    def summarize_table(self, factory, model, table, user_info):
        table_summarization_chain = factory.build_summarization_chain(
            model, self.data_access, table
        )
        table_summarization = table_summarization_chain.invoke(
            {"user_info_not_summary": user_info}
        )
        self.log_response(
            "Table Summary",
            f"Summary of what we know about customer from this table:\n\n {table_summarization}",
        )
        return table_summarization

    async def summarize_table_async(self, factory, model, table, user_info):
        table_summarization_chain = factory.build_summarization_chain(
            model, self.data_access, table
        )
        table_summarization = await table_summarization_chain.ainvoke(
            {"user_info_not_summary": user_info}
        )
        self.log_response(
            "Table Summary",
            f"Summary of what we know about customer from this table:\n\n {table_summarization}",
        )
        return table_summarization

    async def summarize_table_and_generation_questions_async(
        self,
        factory: ChainFactory,
        model,
        model35,
        table,
        user_info: UserInfo,
        customer_question: str,
    ) -> List[str]:
        table_summarization_chain = factory.build_summarization_chain(
            model, self.data_access, table
        )
        question_chain = factory.build_question_generation_chain(
            model35, table_summarization_chain
        )
        questions: List[str] = await question_chain.ainvoke(
            {"user_info_not_summary": user_info, "customer_question": customer_question}
        )  # array of
        self.log_response(
            "Question list",
            f"List of questions the customer might ask:\n\n {questions}",
        )
        return questions

    async def reduce_keywords_async(
        self, model35, factory, table_summarization, keyword_list_slices
    ):
        keyword_reduction_chains = [
            (
                slice_[0],
                factory.build_keyword_reduction_prompt_chain(
                    model35, table_summarization, slice_[1]
                ),
            )
            for slice_ in keyword_list_slices
        ]
        reduced_keyword_list = await asyncio.gather(
            *[
                self._invoke_chain_async(keyed_chain)
                for keyed_chain in keyword_reduction_chains
            ]
        )
        return reduced_keyword_list

    async def _invoke_chain_async(self, keyed_chain):
        key, chain = keyed_chain
        result = await chain.ainvoke({})
        return key, result

    @DeprecationWarning
    def reduce_keywords(
        self, model35, factory, table_summarization, keyword_list_slices
    ):
        keyword_reduction_chains = [
            (
                slice_[0],
                factory.build_keyword_reduction_prompt_chain(
                    model35, table_summarization, slice_[1]
                ),
            )
            for slice_ in keyword_list_slices
        ]
        with ThreadPoolExecutor() as executor:
            reduced_keyword_list = list(
                executor.map(
                    lambda keyed_chain: (keyed_chain[0], keyed_chain[1].invoke({})),
                    keyword_reduction_chains,
                )
            )
        return reduced_keyword_list

    def deduplicate_keywords(self, reduced_keyword_list):
        deduped_segment_keywords = {}
        for key, result in reduced_keyword_list:
            deduped_segment_keywords.setdefault(key, []).extend(result)
        return {
            key: list(set(value)) for key, value in deduped_segment_keywords.items()
        }

    def get_real_keys(self, all_collection_keywords, deduped_segment_keywords):
        real_deduped_keys = {}
        for key, values in deduped_segment_keywords.items():
            real_deduped_keys[key] = [
                value for value in values if value in all_collection_keywords[key]
            ]
        return real_deduped_keys

    @DeprecationWarning
    def identify_predicates(
        self, factory, model, table_summarization, real_deduped_keys
    ):
        predicate_identification_chain = factory.build_collection_predicate_chain(
            model, table_summarization, real_deduped_keys
        )
        return predicate_identification_chain.invoke({})

    async def identify_predicates_async(
        self, factory, model, table_summarization, real_deduped_keys
    ):
        predicate_identification_chain = factory.build_collection_predicate_chain(
            model, table_summarization, real_deduped_keys
        )
        return await predicate_identification_chain.ainvoke({})

    async def summarize_topics_async(
        self,
        collection_predicates,
        table_summarization,
        model,
        factory,
        data_access,
        limit: int,
    ):
        tasks = [
            self.process_topic_summary_async(
                predicate, table_summarization, model, factory, data_access, limit
            )
            for predicate in collection_predicates
        ]
        return await asyncio.gather(*tasks)

    # async def answer_questions_async(
    #     self,
    #     questions: List[str],
    #     model,
    #     factory,
    #     data_access,
    #     limit: int,
    # ):
    #     tasks = [
    #         self.process_question_async(
    #             question, model, factory, data_access, limit
    #         )
    #         for question in questions
    #     ]
    #     return await asyncio.gather(*tasks)

    @DeprecationWarning
    def summarize_topics(
        self, collection_predicates, table_summarization, model, factory, data_access
    ):
        with ThreadPoolExecutor() as executor:
            topic_summaries_for_table = list(
                executor.map(
                    lambda predicate: self.process_topic_summary(
                        predicate, table_summarization, model, factory, data_access
                    ),
                    collection_predicates,
                )
            )
        return topic_summaries_for_table

    # async def process_question_async(
    #     self, question: str, model, factory, data_access, limit: int
    # ):
    #     search_results_for_question = (
    #         await data_access.colbert_manager?.search(
    #             question
    #         )
    #     )
    #     summarization_of_topic_chain = factory.build_vector_search_summarization_chain(
    #         model, search_results_for_question
    #     )
    #     return await summarization_of_topic_chain.ainvoke({})

    async def process_topic_summary_async(
        self, predicate, table_summarization, model, factory, data_access, limit: int
    ):
        search_results_for_topic = (
            await data_access.collection_manager.filtered_ANN_search_async(
                predicate, table_summarization, limit
            )
        )
        summarization_of_topic_chain = factory.build_vector_search_summarization_chain(
            model, search_results_for_topic
        )
        return await summarization_of_topic_chain.ainvoke({})

    @DeprecationWarning
    def process_topic_summary(
        self, predicate, table_summarization, model, factory, data_access
    ):
        search_results_for_topic = data_access.collection_manager.filtered_ANN_search(
            predicate, table_summarization
        )
        summarization_of_topic = factory.build_vector_search_summarization_chain(
            model, search_results_for_topic
        ).invoke({})
        return summarization_of_topic

    @DeprecationWarning
    def summarize_findings(self, factory, model, topic_summaries_for_table):
        topic_summaries_for_table_as_string = json.dumps(topic_summaries_for_table)
        summarization_of_findings_for_table = (
            factory.build_vector_search_summarization_chain(
                model, topic_summaries_for_table_as_string
            )
        )
        return summarization_of_findings_for_table.invoke({})

    async def summarize_findings_async(self, factory, model, topic_summaries_for_table):
        topic_summaries_for_table_as_string = json.dumps(topic_summaries_for_table)
        summarization_of_findings_for_table = (
            factory.build_vector_search_summarization_chain(
                model, topic_summaries_for_table_as_string
            )
        )
        return await summarization_of_findings_for_table.ainvoke({})

    @DeprecationWarning
    def generate_recommendation(
        self, factory, model, all_user_table_summaries, all_table_insights, user_message
    ):
        recommendation_chain = factory.build_final_recommendation_chain_non_parallel(
            model
        )
        recommendation = recommendation_chain.invoke(
            {
                "user_summary": all_user_table_summaries,
                "business_summary": all_table_insights,
                "user_messages": [user_message],
                "our_responses": self.our_responses[::-1],
            }
        )
        return recommendation

    async def generate_recommendation_async(
        self, factory, model, all_user_table_summaries, all_table_insights, user_message
    ):
        recommendation_chain = factory.build_final_recommendation_chain_non_parallel(
            model
        )
        recommendation = await recommendation_chain.ainvoke(
            {
                "user_summary": all_user_table_summaries,
                "business_summary": all_table_insights,
                "user_messages": [user_message],
                "our_responses": self.our_responses[::-1],
            }
        )
        return recommendation
