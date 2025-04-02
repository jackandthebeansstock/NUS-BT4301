import logging
import json
import time
from kafka import KafkaConsumer, errors as kafka_errors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('consumer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_consumer(max_retries=5, retry_interval=5):
    """Create Kafka consumer with retry logic"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            consumer = KafkaConsumer(
                'user_interactions',
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='user_interactions_group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                security_protocol='PLAINTEXT',
                # Properly ordered timeout configurations:
                fetch_max_wait_ms=500,          # Minimum wait time
                request_timeout_ms=30500,       # Must be > fetch_max_wait_ms
                connections_max_idle_ms=31000,  # Must be > request_timeout_ms
                api_version=(2, 6, 0),
                # Additional recommended settings:
                heartbeat_interval_ms=3000,
                session_timeout_ms=10000
            )
            logger.info("Successfully connected to Kafka")
            return consumer
        except kafka_errors.NoBrokersAvailable:
            retry_count += 1
            logger.warning(f"Unable to connect to Kafka (attempt {retry_count}/{max_retries}). Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        except Exception as e:
            logger.error(f"Unexpected error creating consumer: {str(e)}")
            raise
    
    raise ConnectionError(f"Failed to connect to Kafka after {max_retries} attempts")

def consume_messages(consumer):
    """Consume messages from Kafka"""
    try:
        logger.info("Starting to consume messages from 'user_interactions' topic...")
        for message in consumer:
            try:
                data = message.value
                logger.info(f"Received message [Partition: {message.partition}, Offset: {message.offset}]: {data}")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode message: {message.value}")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
    except kafka_errors.KafkaError as e:
        logger.error(f"Kafka error occurred: {str(e)}", exc_info=True)
    finally:
        logger.info("Closing consumer...")
        consumer.close()

if __name__ == "__main__":
    try:
        consumer = create_consumer()
        consume_messages(consumer)
    except Exception as e:
        logger.error(f"Fatal error in consumer: {str(e)}", exc_info=True)
        raise