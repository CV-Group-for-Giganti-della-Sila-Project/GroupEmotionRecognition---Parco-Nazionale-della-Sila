import paho.mqtt.client as mqtt
import time
import threading
import json


'''
#usage example in another script

def main():
    print("Starting example script...")
    
    # set info just once before sending stuff
    # if the next line is blank everything will be sent by default to mosquitto test broker on topic smartpark/telemetry/GSP-EG5120-01
    publisher.set_info("7.tcp.eu.ngrok.io", 12132, "smartpark/telemetry/GSP-EG5120-01") # host, port, topic

    # just need to call this func from the main or endpoint cbk
    publisher.publish("Tree1", 4, "neutral", 12112311)

    # whenever the backend stops, it would be good to call this but not mandatory
    publisher.stop()
    print("connection terminated, program finished")

if __name__ == "__main__":
    main()

'''


class MQTTPublisherService:
    def __init__(self, broker="test.mosquitto.org", port=1883, client_id="EC2Emo", topic = "smartpark/telemetry/GSP-EG5120-01"):
        self.broker = broker
        self.port = port
        self.topic = topic
        
        # If using paho-mqtt v2.0.0+, it's recommended to specify CallbackAPIVersion
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id)
        #self.client = mqtt.Client(client_id)
        
        self.connected = False
        
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
        # Lock to prevent concurrent connections if called from multiple threads
        self._lock = threading.Lock()

    def set_info(self, broker, port, topic):
        self.broker = broker
        self.port = port
        self.topic = topic


    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            print(f"Connected to MQTT broker at {self.broker}")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        self.connected = False
        print("Disconnected from MQTT broker")

    def start(self):
        """Starts the MQTT network loop and connects to the broker."""
        with self._lock:
            if not self.connected:
                try:
                    self.client.connect(self.broker, self.port)
                    # loop_start() creates a background thread to handle MQTT operations automatically
                    self.client.loop_start() 
                    
                    # Wait briefly to allow the connection to be established
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error connecting to MQTT broker: {e}")

    def stop(self):
        """Stops the MQTT network loop and disconnects."""
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, node_id, num_people, g_emotion, timestamp, qos=0, retain=False):
        """Publishes a message to a topic."""
        # Prepare the message
        json_to_send = {
            "thingId": node_id,
            "features": {
                "sensors": {
                    "properties": { 
                    "numPeople": num_people,
                    "gEmotion": g_emotion,
                    "timestamp": timestamp
                    }
                }
            }
        } 

        # Ensure we are connected before trying to publish
        if not self.connected:
            self.start()
        
        if self.connected:
            result = self.client.publish(self.topic, json.dumps(json_to_send), qos, retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published '{json.dumps(json_to_send)}' to topic '{self.topic}'")
                return True
            else:
                print(f"Failed to publish message to topic '{self.topic}'")
                return False
        else:
            print("Cannot publish: Not connected to broker.")
            return False

# Global instance to be imported by other modules.
# It maintains the connection in the background.
publisher = MQTTPublisherService(broker="test.mosquitto.org")



