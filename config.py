import os

scratch_path = os.getenv("SCRATCH_PATH", "scratch")
data_path = os.getenv("DATA_PATH", "data")

os.makedirs(scratch_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
