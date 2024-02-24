from typing import List

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

def create_outlook_ingest_script(
        client_credential: str,
        client_id: str,
        tenant_id: str,
        user_email: str,
        outlook_folders: List[str],
        output_dir: str,
    ):
    """
    Args:
    - client_credentials (str): Client access config credentials (user input).
    - client_id (str): Client ID for connector config (user input).
    - tenant_id (str): Tenant/Org ID for connector config (user input).
    - user_email (str): User email address (user input).
    - outlook_folders (List[str]): List of folders to collect. Ex Sent Items, Inbox, Spam (user input).
    - output_dir (str): Directory to route results to (agent selection).
    """
    
    python_code = f"""
    from unstructured.ingest.connector.outlook import OutlookAccessConfig, SimpleOutlookConfig
    from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
    from unstructured.ingest.runner import OutlookRunner

    if __name__ == "__main__":
        runner = OutlookRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir={output_dir},
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(),
            connector_config=SimpleOutlookConfig(
                access_config=OutlookAccessConfig(
                    client_credential={client_credential},
                ),
                client_id={client_id},
                tenant={tenant_id},
                user_email={user_email},
                outlook_folders={outlook_folders},
                recursive=True,
            ),
        )
        runner.run()
    """
    with open("script_to_run.py", "w") as file:
        file.write(python_code)

def create_salesforce_ingest_script(
        consumer_key: str,
        username: str,
        private_key_path: str,
        categories: List[str],
        output_dir: str,
    ):

    """
    Args:
    - consumer_key (str): Consumer key for access config (user input).
    - username (str): Username for the Salesforce account (user input).
    - private_key_path (str): Path to the account private key (user input).
    - categories (List[str]): Categories to collect. Ex EmailMessage, Account, Lead, Case, Campaign (user input or agent selection)
    - output_dir (str): Directory to route results to (agent selection).
    """

    python_code = f"""
    from unstructured.ingest.connector.salesforce import SalesforceAccessConfig, SimpleSalesforceConfig
    from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
    from unstructured.ingest.runner import SalesforceRunner

    if __name__ == "__main__":
        runner = SalesforceRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir={output_dir},
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(),
            connector_config=SimpleSalesforceConfig(
                access_config=SalesforceAccessConfig(
                    consumer_key={consumer_key},
                ),
                username={username},
                private_key_path={private_key_path},
                categories={categories},
                recursive=True,
            ),
        )
        runner.run()
    """
    with open("script_to_run.py", "w") as file:
        file.write(python_code)


# having to know the exact page title might make this useless
# TODO investigate Searx integration to find the most relevant page name before populating page_title arg
def create_wikipedia_ingest_script(
        page_title: str,        
        output_dir: str,
    ):

    """
    Args:
    - page_title (str): Title of Wikipedia page to search (agent selection).
    - output_dir (str): Directory to route results to (agent selection).
    """
    
    python_code = f"""
    from unstructured.ingest.connector.wikipedia import SimpleWikipediaConfig
    from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
    from unstructured.ingest.runner import WikipediaRunner

    if __name__ == "__main__":
        runner = WikipediaRunner(
            processor_config=ProcessorConfig(
                verbose=True,
                output_dir={output_dir},
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(),
            connector_config=SimpleWikipediaConfig(
                page_title={page_title},
                auto_suggest=False,
            ),
        )
        runner.run()
    """
    with open("script_to_run.py", "w") as file:
        file.write(python_code)