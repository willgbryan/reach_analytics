def create_airtable_ingest_script(
        airtable_pat: str,
        output_dir: str,
    ):

    """
    Args:
    - airtable_pat (str): Airtable Personal Access Token (user input).
    - output_dir (str): Directory to route results to (agent selection).
    """
    
    python_code = f"""
    from unstructured.ingest.connector.airtable import AirtableAccessConfig, SimpleAirtableConfig
    from unstructured.ingest.interfaces import (
        PartitionConfig,
        ProcessorConfig,
        ReadConfig,
    )
    from unstructured.ingest.runner import AirtableRunner

    if __name__ == "__main__":
        runner = AirtableRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir={output_dir},
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(),
            connector_config=SimpleAirtableConfig(
                access_config=AirtableAccessConfig(
                    personal_access_token={airtable_pat}
                ),
            ),
        )
        runner.run()
        """
    with open("script_to_run.py", "w") as file:
        file.write(python_code)

def create_azure_ingest_script(
        account_name: str,
        remote_url: str,
        output_dir: str,
    ):

    """
    Args:
    - account_name (str): Azure account name for Azure access config (user input).
    - remote_url (str): Remote url for blob storage connector config (user input).
    - output_dir (str): Directory to route results to (agent selection).
    """
    
    python_code = f"""
    from unstructured.ingest.connector.fsspec.azure import (
        AzureAccessConfig,
        SimpleAzureBlobStorageConfig,
    )
    from unstructured.ingest.interfaces import (
        PartitionConfig,
        ProcessorConfig,
        ReadConfig,
    )
    from unstructured.ingest.runner import AzureRunner

    if __name__ == "__main__":
        runner = AzureRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir={output_dir},
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(),
            connector_config=SimpleAzureBlobStorageConfig(
                access_config=AzureAccessConfig(
                    account_name={account_name},
                ),
                remote_url={remote_url},
            ),
        )
        runner.run()
    """
    with open("script_to_run.py", "w") as file:
        file.write(python_code)


def create_google_drive_ingest_script(
        service_account_key: str,
        drive_id: str, 
        output_dir: str
    ):

    """
    Args:
    - service_account_key (str): Google Service Account key (user input).
    - drive_id (str): Target file or folder name in Google Drive (user input or agent selection). 
    - output_dir (str): Directory to route results to (agent selection).
    """
    
    python_code = f"""
    from unstructured.ingest.connector.google_drive import (
        GoogleDriveAccessConfig,
        SimpleGoogleDriveConfig,
    )
    from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
    from unstructured.ingest.runner import GoogleDriveRunner

    if __name__ == "__main__":
        runner = GoogleDriveRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir="{output_dir}",
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(),
            connector_config=SimpleGoogleDriveConfig(
                access_config=GoogleDriveAccessConfig(
                    service_account_key="{service_account_key}"
                ),
                recursive=True,
                drive_id="{drive_id}",
            ),
        )
        runner.run()
    """
    with open("script_to_run.py", "w") as file:
        file.write(python_code)
