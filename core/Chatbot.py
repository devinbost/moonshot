import asyncio
import json
import logging
from datetime import datetime
from operator import itemgetter
from typing import List, Dict, Any, Tuple, cast

import PromptFactory
from ChainFactory import ChainFactory
from DataAccess import DataAccess
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from core.CollectionManager import CollectionManager

from core.ConfigLoader import ConfigLoader
from core.EmbeddingFactory import EmbeddingFactory
from core.LLMFactory import LLMFactory
from core.VectorStoreFactory import VectorStoreFactory
from core.adapters.EmbeddingInterface import EmbeddingInterface
from pydantic_models.QAPair import QAPair
from pydantic_models.UserInfo import UserInfo


class Chatbot:
    def __init__(self, config_loader: ConfigLoader, collection_name: str, embedding: EmbeddingInterface,
                 model: ChatOpenAI):
        """
        Initialize the Chatbot with a data access object.
        Parameters:
            data_access (DataAccess): The data access object to interact with the database.
        """
        self.our_responses = [""]
        self.column = None
        self.user_messages = [""]
        print("ran __init__ on Chatbot")
        self.relevant_table_cache = None

        self.llm_factory = LLMFactory(config_loader)
        self.chain_factory = ChainFactory(self.llm_factory)
        self.embedding_factory = EmbeddingFactory(config_loader)
        self.vectorstore_factory = VectorStoreFactory(embedding, config_loader)
        self.astrapydb = self.vectorstore_factory.create_vector_store("AstraPyDB")
        self.collection_manager = CollectionManager(self.astrapydb, embedding, collection_name)
        self.data_access = DataAccess(config_loader, self.vectorstore_factory)
        self.model = model

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
        question_count: str
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
            data_access = (
                self.data_access
            )  # Assuming self.data_access is already instantiated in __init__

            self.log_response("Start", "Inspecting all tables in the database")

            relevant_tables = await self.chain_factory.get_relevant_tables(
                data_access, self.model, user_info
            )
            self.relevant_table_cache = relevant_tables

        all_generated_questions = []

        process_table_futures = [
            self.process_table_async(
                all_generated_questions,
                self.collection_manager,
                self.model,
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
            self.model, all_generated_questions, all_resolved_qa_pairs, user_message
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
        model,
        table,
        user_info: UserInfo,
        summarization_limit: int,
        user_message: str,
            question_count: str
    ) -> List[QAPair]:
        self.log_response("Found Table", f"Found relevant table: {table.table_name}")
        table_summarization = await self.summarize_table_async(
            model, table, user_info
        )
        generated_questions = await self.generate_questions_async(model, table_summarization, user_message,
                                                                  question_count)
        for question in generated_questions:
            all_generated_questions.append(question)

        answers_to_generated_questions = await self.answer_generated_questions_async(
            generated_questions,
            model,
            collection_manager,
            summarization_limit,
        )
        answer_pairs_as_strings = [answerpair.model_dump() for answerpair in answers_to_generated_questions]
        self.log_response("Questions and answers", answer_pairs_as_strings)

        return answers_to_generated_questions

    async def answer_generated_questions_async(
        self,
        generated_questions: List[str],
        model: ChatOpenAI,
        collection_manager: CollectionManager,
        vector_search_limit: int,
    ) -> List[QAPair]:
        """Returns """
        qa_chain = self.chain_factory.build_answer_chain(model)
        # Note that this cast used below is just for the type hint. It's a no-op at runtime.
        answered_question_pairs = cast(List[QAPair], await asyncio.gather(
            *[
                self.chain_factory.invoke_answer_chain(qa_chain, collection_manager, question, vector_search_limit)
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

    async def summarize_table_async(self, model, table, user_info):
        table_summarization_chain = self.chain_factory.build_summarization_chain(
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

    async def generate_questions_async(self, model, table_summary: str, user_message: str,
                                       question_count: str):
        question_generation_chain = self.chain_factory.build_question_generation_chain_from_summary(model)
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
    async def generate_recommendation_async(
        self, model, all_user_table_summaries, all_table_insights, user_message
    ):
        recommendation_chain = self.chain_factory.build_final_recommendation_chain_non_parallel(
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
