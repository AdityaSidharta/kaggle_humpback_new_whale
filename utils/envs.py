import os

project_path = os.getenv("PROJECT_PATH")

data_path = os.path.join(project_path, "data")
output_path = os.path.join(project_path, "output")

model_cp_path = os.path.join(output_path, "model_checkpoint")
logger_path = os.path.join(output_path, "logs")
result_path = os.path.join(output_path, "result")
result_metadata_path = os.path.join(output_path, 'result_metadata')
artifacts_path = os.path.join(output_path, 'artifacts')

logger_repo = os.path.join(logger_path, "logger.log")

train_repo = os.path.join(data_path, "train.csv")
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')