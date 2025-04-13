import logging
import json
import time
from kafka import KafkaConsumer, errors as kafka_errors
import asyncio
import websockets

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

# WebSocket clients
connected_clients = set()

def create_consumer(topics, max_retries=5, retry_interval=5):
    """Create Kafka consumer with retry logic for multiple topics"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=['localhost:9092'],
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='dashboard_group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                security_protocol='PLAINTEXT',
                api_version=(2, 6, 0)
            )
            logger.info(f"Successfully connected to Kafka for topics: {topics}")
            return consumer
        except kafka_errors.NoBrokersAvailable:
            retry_count += 1
            logger.warning(f"Unable to connect to Kafka (attempt {retry_count}/{max_retries}). Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        except Exception as e:
            logger.error(f"Unexpected error creating consumer: {str(e)}")
            raise
    raise ConnectionError(f"Failed to connect to Kafka after {max_retries} attempts")

async def websocket_handler(websocket):
    """Handle WebSocket connections (single argument for newer websockets API)"""
    connected_clients.add(websocket)
    logger.info(f"New WebSocket client connected. Total clients: {len(connected_clients)}")
    try:
        # Keep the connection alive by listening for messages (even if none are sent)
        async for message in websocket:
            logger.debug(f"Received message from client: {message}")
    except websockets.ConnectionClosed:
        logger.info(f"WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"WebSocket handler error: {str(e)}")
    finally:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(connected_clients)}")

async def broadcast_metrics(metrics):
    """Broadcast metrics to all connected WebSocket clients"""
    if connected_clients:
        message = json.dumps(metrics)
        logger.debug(f"Broadcasting metrics to {len(connected_clients)} clients: {message}")
        await asyncio.gather(*(client.send(message) for client in connected_clients))

async def consume_messages(consumer):
    """Consume messages from Kafka and broadcast metrics"""
    logger.info("Starting to consume messages from 'user_interactions' and 'metrics_topic'...")
    while True:
        try:
            msg_pack = consumer.poll(timeout_ms=1000)
            for topic_partition, messages in msg_pack.items():
                for message in messages:
                    try:
                        topic = message.topic
                        data = message.value
                        logger.info(f"Received message from '{topic}' [Partition: {message.partition}, Offset: {message.offset}]: {data}")
                        if topic == 'metrics_topic':
                            await broadcast_metrics(data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode message from '{message.topic}': {message.value}")
                    except Exception as e:
                        logger.error(f"Error processing message from '{message.topic}': {str(e)}", exc_info=True)
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in consume_messages loop: {str(e)}", exc_info=True)
            await asyncio.sleep(1)

async def main():
    """Run Kafka consumer and WebSocket server"""
    topics = ['user_interactions', 'metrics_topic']
    consumer = create_consumer(topics)
    
    # Use the updated handler signature
    websocket_server = await websockets.serve(websocket_handler, "localhost", 8765)
    logger.info("WebSocket server started on ws://localhost:8765")
    
    await consume_messages(consumer)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error in consumer: {str(e)}", exc_info=True)
        raise