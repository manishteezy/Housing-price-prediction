from kafka import KafkaConsumer                                                                                       
from random import randint                                                                                              
from time import sleep                                                                                                  
import sys        
import pandas as pd    

from utils import predict

BROKER = 'kafka:9092'                                                                                               
TOPIC = 'houseData'                                                                                                      
                                                                                                                        
try:                                                                                                                    
    consumer = KafkaConsumer(TOPIC, bootstrap_servers=BROKER)                                                                         
except Exception as e:                                                                                                  
    print(f"ERROR --> {e}")                                                                                             
    sys.exit(1)                                                                      

for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    df = message.value
    print("The prediction for ", df)