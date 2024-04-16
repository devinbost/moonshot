from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from core.ConfigLoader import ConfigLoader
import os

class LLMFactory:

    def __init__(
        self, config_loader: ConfigLoader
    ):
        self.config_loader = config_loader
    def create_llm_model(self, llm_type):
        if llm_type == "watsonx":
            model_id = "google/flan-t5-xxl"
            credentials = {
                "url": "https://us-south.ml.cloud.ibm.com",
                "apikey": self.config_loader.get("llm.watsonx.required.IBM_API_SECRET"),
            }
            gen_parms = {"DECODING_METHOD": "greedy", "MIN_NEW_TOKENS": 1, "MAX_NEW_TOKENS": 50}
            legacy_model = Model(model_id, credentials, gen_parms, self.config_loader.get("llm.watsonx.required.IBM_PROJECT_ID"))
            return WatsonxLLM(model=legacy_model)

        elif llm_type == "openai":
            api_key = self.config_loader.get("llm.openai.required.OPENAI_API_KEY")
            return ChatOpenAI(api_key=api_key, temperature=0, model_name=self.config_loader.get("llm.openai.required.MODEL_NAME"))

        elif llm_type == "openai35":
            api_key = self.config_loader.get("llm.openai.required.OPENAI_API_KEY")
            return ChatOpenAI(api_key=api_key, temperature=0, model_name="gpt-3.5-turbo-0125")

        elif llm_type == "azure":
            key = self.config_loader.get("llm.azure.required.AZURE_OPENAI_API_KEY")
            return AzureChatOpenAI(api_key=key,
                                   azure_endpoint=self.config_loader.get("llm.azure.required.AZURE_OPENAI_ENDPOINT"),
                                   temperature=0,
                                   openai_api_version=self.config_loader.get("llm.azure.required.OPENAI_API_VERSION"),
                                   azure_deployment=self.config_loader.get("llm.azure.required.AZURE_DEPLOYMENT"))

        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

