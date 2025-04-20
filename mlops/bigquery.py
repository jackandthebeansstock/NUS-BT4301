
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions

def list_tables(client, project_id, dataset_id):
    """List all tables in the specified dataset."""
    try:
        dataset_ref = client.dataset(dataset_id, project=project_id)
        tables = client.list_tables(dataset_ref)
        print(f"Tables in dataset {project_id}.{dataset_id}:")
        table_count = 0
        for table in tables:
            print(table.table_id)
            table_count += 1
        if table_count == 0:
            print("No tables found in the dataset.")
        return table_count
    except exceptions.NotFound:
        print(f"Dataset {project_id}.{dataset_id} not found.")
        return 0
    except exceptions.Forbidden:
        print("Access denied. Ensure the service account has BigQuery permissions.")
        return 0

def read_table(client, project_id, dataset_id, table_name):
    """Read data from the specified table."""
    try:
        # Define the query
        query = f"""
            SELECT *
            FROM `{project_id}.{dataset_id}.{table_name}`
            LIMIT 100
        """
        # Run the query
        query_job = client.query(query)
        # Fetch results
        results = query_job.result()
        # Process results
        print(f"Data from {project_id}.{dataset_id}.{table_name}:")
        for row in results:
            print(dict(row))
    except exceptions.NotFound:
        print(f"Table {project_id}.{dataset_id}.{table_name} not found.")
    except exceptions.Forbidden:
        print("Access denied. Ensure the service account has BigQuery permissions.")
    except Exception as e:
        print(f"An error occurred while reading the table: {str(e)}")

def main():
    # Configuration
    KEY_PATH = 'big-query-key.json'  # Path to your service account key file
    PROJECT_ID = 'bt4301-454516'    # Your project ID
    DATASET_ID = 'kindle_store'      # Your dataset
    TABLE_NAME = 'metadata'   # Replace with the table name after listing tables

    try:
        # Create credentials
        credentials = service_account.Credentials.from_service_account_file(
            KEY_PATH,
            scopes=['https://www.googleapis.com/auth/cloud-platform'],
        )

        # Initialize BigQuery client
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

        # Step 1: List tables in the dataset
        print("Listing tables...")
        table_count = list_tables(client, PROJECT_ID, DATASET_ID)
        
        if table_count == 0:
            print("Cannot proceed with reading data due to no tables or dataset issues.")
            return

        # Step 2: Read data from the specified table
        print(f"\nReading data from table: {TABLE_NAME}")
        read_table(client, PROJECT_ID, DATASET_ID, TABLE_NAME)

    except FileNotFoundError:
        print(f"Service account key file not found at {KEY_PATH}. Please verify the file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
