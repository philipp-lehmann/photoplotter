import paho.mqtt.client as mqtt
import time
import random
import os

class StateEngine:
    def __init__(self):
        # State
        self.state = "Template"
        self.reprint = False
        self.currentPhotoPath = ""
        self.currentWorkPath = ""
        self.currentSVGPath = ""
        self.imagesPerRow = 5
        self.imagesPerColumn = 3
        self.totalImages = self.imagesPerColumn * self.imagesPerRow
        self.paperSizeX = 1587 #1191 multiplied by higher 96 dpi of Nextdraw
        self.paperSizeY = 1122 #841 multiplied by higher 96 dpi of Nextdraw
        self.workID = 0
        self.photoID = list(range(1, self.totalImages + 1))  # List of positions from 1 to totalImages
        self.transitions = {
            "Startup": ["Waiting", "ResetPending", "Test"],
            "Waiting": ["Tracking"],
            "Tracking": ["Working", "Snapping", "Tracking"],
            "Working": ["Tracking"],
            "Snapping": ["Tracking", "Processing"],
            "Processing": ["Drawing", "Waiting"],
            "Drawing": ["Waiting", "ResetPending"],
            "ResetPending": ["Waiting", "Template"],
            "Template": ["ResetPending"],
            "Test": ["Waiting", "Drawing"]
        }

        if self.is_running_on_raspberry_pi():
            print(f"\033[1;33m📱 Raspberry Pi found: Connecting to broker\033[0m.")
            self.broker_address = "localhost"
            self.client = mqtt.Client("StateEngine_Client")
            self.client.on_connect = self.on_connect
            self.client.connect(self.broker_address)
            self.client.subscribe("lcd/buttons")
            self.client.loop_start()
            self.client.on_message = self.on_message
            print("MQTT broker started.")
        else:
            print(f"\033[1;32m🖥️ Raspberry Pi not found: Entering test mode\033[0m.")
            self.state = "Test"
        
        print("Starting StateEngine ...")
        self.reset_photo_id()  # Shuffle the photoID list on startup
    
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
                message = f"{new_state}-{self.photoID[-1]}"  # Display the current last photoID
                self.publish_message("state_engine/state", message)
            else:
                self.publish_message("state_engine/state", new_state)
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")

    def update_image_path(self, photo_path):
        self.currentPhotoPath = photo_path
        print(f"Photo path updated to {photo_path}.")
        self.publish_message("state_engine/photo_path", photo_path)
            
    def update_photo_id(self):
        if self.photoID:
            removed_id = self.photoID.pop()  # Remove the last photoID
            print(f"Photo ID removed: {removed_id}, remaining IDs: {self.photoID}")
            if not self.photoID:  # If the list becomes empty, trigger reset
                print("All images processed. Transitioning to ResetPending.")
                self.change_state("ResetPending")
        else:
            print("No photo IDs available.")

    def reset_photo_id(self):
        triplets = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]
        for triplet in triplets:
            random.shuffle(triplet)
        self.photoID = [item for triplet in triplets for item in triplet]
        print(f"Photo IDs reset and shuffled within triplets: {self.photoID}")
    
    def update_work_id(self):
        self.workID += 1
        print(f"Work ID: {self.workID}")
        
    def reset_work_id(self):
        print(f"Reset Work ID: {self.workID} -> 0")
        self.workID = 0
      
    def get_image_params_by_id(self, id=0):
        # Define border and gutter
        borderSize = 50
        gutterSize = 50

        # Adjusted maximum dimensions to account for border
        maxX = self.paperSizeX - (borderSize * 2)  # Maximum X dimension in mm for A3 paper
        maxY = self.paperSizeY - (borderSize * 2)  # Maximum Y dimension in mm for A3 paper

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
        # Manually handle reset
        message = msg.payload.decode()
        reset_keys = ["KEY2", "KEY3", "UP", "DOWN", "LEFT", "RIGHT"]
        template_keys = ["KEY1"]
        
        # Handle messages received on subscribed topics
        print(f"Received message on topic '{msg.topic}': {message}")

        if self.state == "ResetPending":
            if message in reset_keys:
                print("Reset confirmed")
                self.reset_photo_id()  # Reset and shuffle photo IDs
                self.change_state("Waiting")
                time.sleep(1)    
            elif message in template_keys:
                print("Applying template")
                self.change_state("Template")
                time.sleep(1)
        
        # Handler for each state
        elif self.state == "Waiting":
            self.reset_work_id()
            self.change_state("Tracking")
        elif self.state == "Working":
            self.change_state("Tracking")
        elif self.state == "Printing":
            self.reprint = True
        else:
            print(f"Unexpected state: {self.state}") 
        
    def publish_message(self, topic, message):
        self.client.publish(topic, message)
        
        
    # OS
    # ------------------------------------------------------------------------ 
    @staticmethod
    def is_running_on_raspberry_pi():
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "Raspberry Pi" in cpuinfo
        except FileNotFoundError:
            return False
