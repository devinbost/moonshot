import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Any, Tuple, cast

from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1.json import pydantic_encoder

import PromptFactory
from ChainFactory import ChainFactory
from DataAccess import DataAccess
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
from core.LLMFactory import LLMFactory
from core.VectorStoreFactory import VectorStoreFactory
from pydantic_models.QAPair import QAPair
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
        self.data_access = data_access
        self.relevant_table_cache = None

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
        question_count: str,
        collection_name: str,
        embedding_model: str
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
            # Moved the instantiation of models and factories into if-block to avoid unnecessary recreations

            config_loader = ConfigLoader()
            llm_factory = LLMFactory(config_loader)
            chain_factory = ChainFactory(llm_factory)
            embedding_manager = EmbeddingManager(model_name=embedding_model) # TODO: pull this up so people can change it
            vectorstore_factory = VectorStoreFactory(embedding_manager, config_loader)
            astrapydb = vectorstore_factory.create_vector_store("AstraPyDB")
            collection_manager = CollectionManager(astrapydb, embedding_manager, collection_name)
            model = llm_factory.create_llm_model("openai",model_name="gpt-4-1106-preview")

            model35 = llm_factory.create_llm_model("openai",model_name="gpt-3.5-turbo-1106")
            data_access = (
                self.data_access
            )  # Assuming self.data_access is already instantiated in __init__

            self.log_response("Start", "Inspecting hundreds of tables in the database")

            relevant_tables = await chain_factory.get_relevant_tables(
                data_access, model, user_info
            )
            self.relevant_table_cache = relevant_tables

        all_generated_questions = []

        process_table_futures = [
            self.process_table_async(
                all_generated_questions,
                collection_manager,
                chain_factory,
                model,
                model35,
                table,
                user_info,
                summarization_limit,
                user_message,
                question_count
            )
            for table in self.relevant_table_cache
        ]

        all_resolved_qa_pairs = await asyncio.gather(*process_table_futures)

        recommendation = await self.generate_recommendation_async(
            chain_factory, model, all_generated_questions, all_resolved_qa_pairs, user_message
        )
        self.our_responses.append(recommendation)
        self.log_response(
            "Recommendation", f"Recommendation to the user is: {recommendation}"
        )
        bot_chat_area = col1.markdown(recommendation)
        return recommendation

    async def process_table_async(
        self,
        all_generated_questions,
        collection_manager: CollectionManager,
        chain_factory: ChainFactory,
        model,
        model35,
        table,
        user_info: UserInfo,
        summarization_limit: int,
        user_message: str,
            question_count: str
    ) -> List[QAPair]:
        self.log_response("Found Table", f"Found relevant table: {table.table_name}")
        table_summarization = await self.summarize_table_async(
            chain_factory, model, table, user_info
        )
        generated_questions = await self.generate_questions_async(chain_factory, model, table_summarization, user_message,
                                                                  question_count)
        for question in generated_questions:
            all_generated_questions.append(question) # Perhaps we should instead append generated_questions directly instead of this loop.

        answers_to_generated_questions = await self.answer_generated_questions_async(
            generated_questions,
            model,
            chain_factory,
            collection_manager,
            summarization_limit,
        )
        answer_pairs_as_strings = [answerpair.model_dump() for answerpair in answers_to_generated_questions]
        self.log_response("Questions and answers", answer_pairs_as_strings)

        # insights_on_table = await self.summarize_findings_async(
        #     factory, model, topic_summaries_for_table
        # )
        return answers_to_generated_questions

    async def answer_generated_questions_async(
        self,
        generated_questions: List[str],
        model: ChatOpenAI,
        chain_factory: ChainFactory,
        collection_manager: CollectionManager,
        vector_search_limit: int,
    ) -> List[QAPair]:
        """Returns """
        qa_chain = chain_factory.build_answer_chain(model)
        # Note that this cast used below is just for the type hint. It's a no-op at runtime.
        answered_question_pairs = cast(List[QAPair], await asyncio.gather(
            *[
                chain_factory.invoke_answer_chain(qa_chain, collection_manager, question, vector_search_limit)
                for question in generated_questions
            ]
        ))
        return answered_question_pairs

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

    async def generate_questions_async(self, factory: ChainFactory, model, table_summary: str, user_message: str,
                                       question_count: str):
        question_generation_chain = factory.build_question_generation_chain_from_summary(model, question_count)
        questions = await question_generation_chain.ainvoke(
            {"table_summary": table_summary,
             "user_message": user_message,
             "question_count": question_count}
        )
        self.log_response(
            "Questions",
            f"Here are some questions we think might be relevant for the user:\n\n {questions}",
        )
        return questions
    async def summarize_table_and_generate_questions_async(self, factory: ChainFactory, model, table, user_info,
                                                           user_message: str, question_count: str):
        """This method version is for if we want to use a larger LCEL chain rather than breaking up the steps."""
        table_summarization_chain = factory.build_summarization_chain(
            model, self.data_access, table
        )
        question_generation_chain = factory.build_question_generation_chain(model, table_summarization_chain)
        questions = await question_generation_chain.ainvoke(
            {"user_info_not_summary": user_info,
             "user_message": user_message,
             "question_count": question_count}
        )
        self.log_response(
            "Questions",
            f"Here are some questions we think might be relevant for the user:\n\n {questions}",
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

    async def _invoke_chain_async(self, chain):
        key, chain = chain
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

    @DeprecationWarning
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

    async def summarize_answers(self, factory, model, qa_pairs: List[QAPair]):
        answers = [qa_pair.answer for qa_pair in qa_pairs]
        json_list_of_answers = json.dumps(answers)
        summarization_of_findings_for_table = (
            factory.build_vector_search_summarization_chain(
                model, json_list_of_answers
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
