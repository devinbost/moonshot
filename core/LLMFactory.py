from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_community.chat_models.openai import ChatOpenAI

from core.ConfigLoader import ConfigLoader


class LLMFactory:

    def __init__(
        self, config_loader: ConfigLoader
    ):
        self.config_loader = config_loader
    def create_llm_model(self, llm_type, **kwargs):

        if llm_type == "watsonx":
            model_id = "google/flan-t5-xxl"
            credentials = {
                "url": "https://us-south.ml.cloud.ibm.com",
                "apikey": self.config_loader.get("ibm_api_secret"),
            }

            gen_parms = {
                "DECODING_METHOD": "greedy",
                "MIN_NEW_TOKENS": 1,
                "MAX_NEW_TOKENS": 50,
            }

            # I occasionally get a 443 CA error that appears to be intermittent. Need exponential backoff/retry.
            legacy_model = Model(
                model_id, credentials, gen_parms, self.config_loader.get("ibm_project_id")
            )
            return WatsonxLLM(model=legacy_model)
        elif llm_type == "openai":
            return ChatOpenAI(temperature=0, model_name=kwargs.get("model_name"))
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
