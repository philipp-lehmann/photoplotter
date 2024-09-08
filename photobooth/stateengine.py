import paho.mqtt.client as mqtt
import time

class StateEngine:
    def __init__(self):
        # State
        self.debugmode = False
        self.state = "Startup"
        self.currentPhotoPath = ""
        self.currentWorkPath = ""
        self.currentSVGPath = ""
        self.imagesPerRow = 5
        self.imagesPerColumn = 3
        self.totalImages = self.imagesPerColumn * self.imagesPerRow
        self.photoID = self.totalImages
        self.transitions = {
            "Startup": ["Waiting", "Test"],
            "Waiting": ["Working"],
            "Working": ["Tracking"],
            "Tracking": ["Processing", "Working"],
            "Processing": ["Drawing"],
            "Drawing": ["Waiting", "ResetPending"],
            "ResetPending": ["Waiting"], 
            "Test": ["Waiting", "Drawing"]
        }
        
        # MQTT
        self.broker_address = "localhost"
        self.client = mqtt.Client("StateEngine_Client")
        self.client.on_connect = self.on_connect
            
        if self.debugmode:
            print("Starting Debugmode...")
            self.photoID = 1
            pass
        else:
            self.client.connect(self.broker_address) 
            self.client.subscribe("lcd/buttons")
            self.client.loop_start()
            self.client.on_message = self.on_message
        
        print("Starting StateEngine ...")
        
    
    # State
    # ------------------------------------------------------------------------     
    def get_state(self):
        return self.state
    
    def change_state(self, new_state):
        # Only update state if transition is possible
        if new_state in self.transitions[self.state]:
            print(f"State change from {self.state} to {new_state}.")
            self.state = new_state
            if self.state == "Drawing":
                # Special case to display the current drawing image 
                message = f"{new_state}-{self.photoID}"
                self.publish_message("state_engine/state", message)
            else:
                self.publish_message("state_engine/state", new_state)
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")

    def update_image_path(self, photo_path):
        self.currentPhotoPath = photo_path
        print(f"Photo path updated to {photo_path}.")
        self.publish_message("state_engine/photo_path", photo_path)
            
    def update_image_id(self):
        self.photoID -= 1
        print(f"Photo ID: {self.photoID}")
    
    def reset_photo_id(self):
        self.photoID = self.totalImages
            
    def get_image_params_by_id(self, id=0):
        # Define border and gutter
        borderSize = 50
        gutterSize = 50

        # Adjusted maximum dimensions to account for border
        paperSizeX = 1587
        paperSizeY = 1122
        maxX = paperSizeX - (borderSize * 2)  # Maximum X dimension in mm for A3 paper
        maxY = paperSizeY - (borderSize * 2)  # Maximum Y dimension in mm for A3 paper

        # Total rows needed, given the total images and images per row
        totalRows = (self.totalImages + self.imagesPerRow - 1) // self.imagesPerRow
        
        # Adjust drawableWidth and drawableHeight to account for gutters
        drawableWidth = maxX - (gutterSize * (self.imagesPerRow - 1))
        drawableHeight = maxY - (gutterSize * (totalRows - 1))

        # Calculate offset for each image, including gutters
        offsetX = drawableWidth / self.imagesPerRow
        offsetY = drawableHeight / totalRows  # Now based on actual totalRows

        # Calculate starting position for the drawing, including border
        col = id % self.imagesPerRow
        row = id // self.imagesPerRow  # Fixed to divide by imagesPerRow for consistency
        
        startPositionX = (col * offsetX) + (col * gutterSize) + borderSize
        startPositionY = (row * offsetY) + (row * gutterSize) + borderSize
        
        return (startPositionX, startPositionY)


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
            self.change_state("Working")
        elif self.state == "Working":
            self.change_state("Tracking")
        elif self.state == "ResetPending":
            print("Reset confirmed")
            self.reset_photo_id()
            self.change_state("Waiting")
            time.sleep(3)
        else:
            pass
        
    def publish_message(self, topic, message):
        self.client.publish(topic, message)

        