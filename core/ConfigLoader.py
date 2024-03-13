import os


class ConfigLoader:
    def __init__(self):
        self.config = {
            # "keyspace": os.getenv("KEYSPACE_NAME", None),
            # "table_name": os.getenv("TABLE_NAME", "default_table_name"),
            # "database_name": os.getenv("DATABASE_NAME", None),
            "token": os.getenv("ASTRA_TOKEN", None),
            "api_endpoint": os.getenv("ASTRA_ENDPOINT", None),
            "secure_bundle_path": os.getenv("SECURE_BUNDLE_PATH", None),
            "ibm_api_secret": os.getenv("IBM_API_SECRET", "not defined. See ConfigLoader.py"),
            "ibm_project_id": os.getenv("IBM_PROJECT_ID", "not defined. See ConfigLoader.py")
        }
        self._validate_config()

    def _validate_config(self):
        for key, value in self.config.items():
            if value is None:
                raise ValueError(f"Environment variable for {key} is not set.")

    def get(self, key):
        return self.config.get(key)
