import os
import yaml

class ConfigLoader:
    """Loads and validates configuration from environment variables, including optional sections."""

    def __init__(self):
        # Initialize any necessary attributes here.
        self.config = None
        self.load_and_validate_config("config/config.yaml")

    def load_and_validate_config(self, yaml_file_path):
        """
        Example usage:
            config_loader.load_and_validate_config('path/to/your/config.yaml')
        """
        with open(yaml_file_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Process LLM configurations under the "llm" top-level property
        if 'llm' in self.config:
            for provider, details in self.config['llm'].items():
                self.process_config_section(provider, details)

        # Process other top-level configurations like LangSmith and embeddings directly
        for section in ['langsmith', 'embedding']:
            if section in self.config:
                self.process_config_section(section, self.config[section])

    def process_config_section(self, section_name, section_details):
        required_vars = section_details.get("required", {})
        optional_vars = section_details.get("optional", {})

        # Validate and set required variables
        for var, value in required_vars.items():
            if value is None:
                raise ValueError(f"Missing required configuration for {var} in section {section_name}")
            os.environ[var] = value

        # Set optional variables if present
        for var, value in optional_vars.items():
            if value is not None:
                os.environ[var] = value

    def get_llm_names(self):
        """Returns a list of the LLM names directly from the YAML configuration."""
        if 'llm' in self.config:
            return list(self.config['llm'].keys())
        return []

    def get_embedding_names(self):
        """Returns a list of the embedding names directly from the YAML configuration."""
        if 'embedding' in self.config:
            return list(self.config['embedding'].keys())
        return []

    def get(self, path):
        """
        Retrieves a specific configuration value based on a given path.
        :param path: A string representing the path to the configuration, using '.' as a delimiter for nested items.
        :return: The configuration value or None if not found.

        Example usage:
            watsonx_config = config_loader.get_config('llm.watsonx')
            ibm_api_secret = config_loader.get_config('llm.watsonx.required.IBM_API_SECRET')
        """
        keys = path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, None)
            if value is None:
                break
        return value
