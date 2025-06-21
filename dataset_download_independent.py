import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_dataset(dataset_name = '', data_path = './data'):
    project_path = os.getcwd()
    os.environ['KAGGLE_CONFIG_DIR'] = project_path

    kaggle_json_path = os.path.join(project_path, 'kaggle.json')
    if not os.path.exists(kaggle_json_path):
        print(f"Warning: kaggle.json not found at {kaggle_json_path}")
        print("Please place your kaggle.json file in the project directory")
        exit(1)
    else:
        print(f"Found kaggle.json at {kaggle_json_path}")

    api = KaggleApi()

    try:
        api.authenticate()
        print("Kaggle API authenticated successfully.")

        print(f"Downloading dataset: {dataset_name} to {data_path}")
        api.dataset_download_files(dataset_name, path=data_path, unzip=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Kaggle API authentication failed. Please check your credentials.")

if __name__ == "__main__":
    dataset_name = 'abhiramasdf/emergency-vehicle-sirens-with-traffic-noise'
    data_path = './other_data'
    download_kaggle_dataset(dataset_name, data_path)