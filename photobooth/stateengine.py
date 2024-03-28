import paho.mqtt.client as mqtt

class StateEngine:
    def __init__(self):
        # State
        self.state = "Startup"
        self.currentPhotoPath = ""
        self.photoID = 0
        self.imagesPerRow = 5
        self.imagesPerColumn = 3
        self.transitions = {
            "Startup": ["Waiting"],
            "Waiting": ["Tracking"],
            "Tracking": ["Processing"],
            "Processing": ["Drawing", "Waiting"],
            "Drawing": ["Waiting"],
            "ResetPending": ["Waiting"]
        }
        
        # MQTT
        self.broker_address = "localhost"  # Assuming Mosquitto is running on the same device
        self.client = mqtt.Client("StateEngine_Client")  # Create a new instance with a unique client ID
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_address) 
        self.client.subscribe("lcd/buttons")
        self.client.loop_start()
        self.client.on_message = self.on_message
        print("Starting StateEngine ...")
        
    
    # Statee
    # ------------------------------------------------------------------------     
    def get_state(self):
        return self.state
    
    def change_state(self, new_state):
        # Only update state if transition is possible
        if new_state in self.transitions[self.state]:
            print(f"State change from {self.state} to {new_state}.")
            self.state = new_state
            self.publish_message("state_engine/state", new_state)
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")

    def update_image_path(self, photo_path):
        self.currentPhotoPath = photo_path
        print(f"Photo path updated to {photo_path}.")
        self.publish_message("state_engine/photo_path", photo_path)
            
    def update_image_id(self):
        self.photoID += 1
        max_images = self.imagesPerRow * self.imagesPerColumn
        
        # Check if all available spots for images have been drawn
        if self.photoID >= max_images:
            self.photoID = 0  
            self.change_state("ResetPending") 
            print(f"photoID reached maximum capacity ({max_images}). Resetting ID and changing state to 'ResetPending'.")


    # Messages
    # ------------------------------------------------------------------------
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
        else:
            print(f"Failed to connect to MQTT broker with error code {rc}")
            
    def on_message(self, client, userdata, msg):
        # Handle messages received on subscribed topics
        print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")
        
        # Handler for each state
        if self.state == "Waiting":
            self.change_state("Tracking")
        elif self.state == "ResetPending":
            self.change_state("Waiting")
        else:
            pass
        
    def publish_message(self, topic, message):
        self.client.publish(topic, message)

        