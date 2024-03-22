from .display import Display

class StateEngine:
    def __init__(self):
        print("Starting StateEngine ...")
        self.display = Display()
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

    def get_state(self):
        return self.state
    
    def change_state(self, new_state):
        # Only update state ii transitions is possible
        if new_state in self.transitions[self.state]:
            print(f"State change from {self.state} to {new_state}.")
            self.state = new_state 
            self.display.update_status(new_state)  
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")
            
    def update_image_id(self):
        self.photoID += 1
        max_images = self.imagesPerRow * self.imagesPerColumn
        
        # Check if all available spots for images have been drawn
        if self.photoID >= max_images:
            self.photoID = 0  
            self.change_state("ResetPending") 
            print("photoID reached maximum capacity ({max_images}). Resetting ID and changing state to 'ResetPending'.")
