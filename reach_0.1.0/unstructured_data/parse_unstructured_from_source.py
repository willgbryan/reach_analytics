def create_google_drive_script(
        service_account_key: str,
        drive_id: str, 
        output_dir: str
        ):
    
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
