import os
import logging


class Config:
    def __init__(self):
        self.scratch_path = os.getenv("SCRATCH_PATH", "scratch")
        self.data_path = os.getenv("DATA_PATH", "data")
        self.openai_json = os.getenv(
            "OPENAI_JSON", "scratch/devin.bost@datastax.com-token.json"
        )
        self.keyspace_name = os.getenv("KEYSPACE_NAME", "keyspace")
        self.table_name = os.getenv("TABLE_NAME", "table")
        self.database_name = os.getenv("DATABASE_NAME", "database")
        self.create_directories()

    def create_directories(self):
        os.makedirs(self.scratch_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

    def secure_connect_bundle_path(self):
        return os.path.join(
            self.scratch_path, f"secure-connect-{self.database_name}.zip"
        )

    def get_secure_bundle_full_path(self):
        return os.path.join(os.getcwd(), self.secure_connect_bundle_path())


config = Config()
logging.basicConfig(
    filename=config.scratch_path + "/crawler.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
