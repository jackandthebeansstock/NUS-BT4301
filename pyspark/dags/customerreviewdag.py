
from airflow.decorators import dag, task
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

@dag(
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
)
def customer_review_dag():
    read_data = SparkSubmitOperator(
        task_id="read_data",
        application="./include/scripts/preprocess_customers.py",
        conn_id="spark",
        verbose=True, #get information if something goes wrong
    )
    
    read_data
    
    # @task()
    # def extract():
    #     print("Extracting data from source")

    # @task()
    # def transform():
    #     print("Transforming data")

    # @task()
    # def load():
    #     print("Loading data to destination")

    # extract >> transform >> load

customer_review_dag()



        # for step in range(data[one_route].values()):
        #     time = datetime.fromisoformat(step['time'])
        #     year = time.year
        #     day_type = 1 if dt.weekday() >= 5 else 0 #weekend is 1 else weekday is 0
        #     seasonality = get_seasonality(time)
        #     time_of_day = time.hour
        #     lat = step['lat']
        #     long = step['long']
        #     bearing = step['bearing']
        #     velocity = step['velocity']
        #     occupancy = step['occupancy']