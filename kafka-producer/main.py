from kafka import KafkaProducer                                                                                         
from random import randint                                                                                              
from time import sleep                                                                                                  
import sys        
import pandas as pd    
import time
import random                                                                                                  
                                                                                                                        
BROKER = 'kafka:9092'                                                                                               
TOPIC = 'houseData'                                                                                                      
                                                                                                                        
test_data = pd.read_csv('./data/test.csv')                                                                       
                                                                                                                        
try:                                                                                                                    
    producer = KafkaProducer(bootstrap_servers=BROKER)                                                                         
except Exception as e:                                                                                                  
    print(f"ERROR --> {e}")                                                                                             
    sys.exit(1)                                                                                                        

for i in range(test_data.shape[0]):
    record = test_data.iloc[i].to_dict()
    print("sending...", str(record))
    producer.send(TOPIC, str(record).encode("utf-8"))
    time.sleep(random.randint(1,4))

