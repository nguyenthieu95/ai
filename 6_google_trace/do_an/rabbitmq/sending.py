import pika

# 1. Tao ket noi den RabbitMq server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 2. Dam bao rang ton` tai queue trong RabbitMQ
channel.queue_declare(queue='hello')        # Tao queue co ten la hello

# 3. Messages body chay qua exchange component de xac dinh xem se~ den queue na`o
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Hello World!')

# 4. In ra tin nhan da~ gui di cai gi
print(" [x] Sent 'Hello World!'")

# 5. Dong' ket noi
connection.close()



