import logging
import json
import pika
from pika import adapters, BlockingConnection, connection


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class RabbitMQClient():
    def __init__(self, host="127.0.0.1", port=5672, username="guest", password="guest", vhost='platform_staging', auto_delete=None,
                 exchange_type='direct'):
        """

        :param host:
        :param port:
        :param username:
        :param password:

        :param vhost:
        """
        self.bk_logger = LOGGER
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.vhost = vhost
        self.auto_delete = auto_delete
        self.exchange_type = exchange_type

        if vhost is None:
            vhost = ''

        params = pika.ConnectionParameters(host=self.host, port=self.port, virtual_host=self.vhost,
                                           credentials=pika.PlainCredentials(username=self.username,
                                                                             password=self.password),
                                           heartbeat=600,
                                           blocked_connection_timeout=300)

        self.connection = BlockingConnection(params)
        self.channel = self.connection.channel()

    def _callback_get_body(self, ch, method, properties, body):
        """

        :param ch:
        :param method:
        :param properties:
        :param body:
        :return:
        """
        return body

    def get_message(self, queue_name=None, no_ack=False, is_json_encode=True):
        """

        :param queue_name:
        :param no_ack:
        :param is_json_encode:
        :return:
        """
        channel = self.connection.channel()
        method_frame, header_frame, body = channel.basic_get(queue=queue_name, no_ack=no_ack)
        if method_frame:
            if not no_ack:
                channel.basic_ack(method_frame.delivery_tag)
        else:
            self.bk_logger.info('No message returned')
        try:
            body = body.decode('utf-8')
            body = json.loads(body)
        except Exception as e:
            self.bk_logger.error(e)
        return body

    def publish_message(self, exchange_name=None, routing_key='', body=None, durable=True, force_mode="rabbitmq"):
        """

        :param exchange_name: string
        :param routing_key: string
        :param body: dict
        :return:
        """

        body = body or " "
        self.channel.exchange_declare(exchange_name, durable=durable, auto_delete=self.auto_delete,
                                      exchange_type=self.exchange_type)

        try:
            if isinstance(body, dict):
                body = json.dumps(body)
        except TypeError as e:
            raise TypeError("Cannot convert to string", e)
        if len(body) == 0:
            raise BufferError("Empty data")

        if force_mode == "direct":
            body["mode"] = "direct"

        # self.bk_logger.info("Publish message to %s", exchange_name)
        result = self.channel.basic_publish(exchange=exchange_name,
                                            routing_key=routing_key,
                                            body=body)
        if result:
            # self.bk_logger.info("Successful!")
            return True
        self.bk_logger.error("Error to publish")
        return False

    def close_channel(self):
        # self.bk_logger.info("Closing channel")
        self.channel.close()