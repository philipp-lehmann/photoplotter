import os

class StateEngine:
    def __init__(self):
        print("Starting StateEngine ...")
        self.state = "Startup"
        self.currentPhotoPath = ""
        self.photoID = 0
        self.imagesPerRow = 5
        self.imagesPerColumn = 3
        self.transitions = {
            "Startup": ["Ready"],
            "Ready": ["Tracking"],
            "Tracking": ["Processing"],
            "Processing": ["Drawing", "Ready"],
            "Drawing": ["Ready"],
            "ResetPending": ["Ready"]
        }
        # Set the path to the FIFO
        self.fifo_path = 'lcd/tmp/state_fifo'

    def get_state(self):
        return self.state
    
    def change_state(self, new_state):
        # Only update state if transition is possible
        if new_state in self.transitions[self.state]:
            print(f"State change from {self.state} to {new_state}.")
            self.state = new_state
            self.write_to_fifo(f"{new_state}.jpg")
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")

    def update_image_path(self, photo_path):
        self.currentPhotoPath = photo_path
        print(f"Photo path updated to {photo_path}.")
        self.write_to_fifo(photo_path)
            
    def update_image_id(self):
        self.photoID += 1
        max_images = self.imagesPerRow * self.imagesPerColumn
        
        # Check if all available spots for images have been drawn
        if self.photoID >= max_images:
            self.photoID = 0  
            self.change_state("ResetPending") 
            print(f"photoID reached maximum capacity ({max_images}). Resetting ID and changing state to 'ResetPending'.")

    def write_to_fifo(self, message):
        # Ensure the FIFO exists before trying to write
        if os.path.exists(self.fifo_path):
            with open(self.fifo_path, 'w') as fifo:
                fifo.write(message + '\n')
        else:
            print(f"Error: FIFO at {self.fifo_path} not found.")
