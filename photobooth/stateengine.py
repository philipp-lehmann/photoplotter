from .display import Display

class StateEngine:
    def __init__(self):
        print("Starting StateEngine ...")
        self.display = Display()
        self.state = "Startup"
        self.imageID = 0
        self.imagesPerRow = 5
        self.imagesPerColumn = 3
        self.transitions = {
            "Startup": ["Ready"],
            "Ready": ["Tracking", "Processing"],
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
            print("State change from {self.state} to {new_state}.")
            self.state = new_state 
            self.display.update_status(new_state)  
        else:
            print("Invalid transition from {self.state} to {new_state}.")
            
    def update_image_id(self):
        self.imageID += 1
        max_images = self.imagesPerRow * self.imagesPerColumn
        
        # Check if all available spots for images have been drawn
        if self.imageID >= max_images:
            self.imageID = 0  
            self.change_state("ResetPending") 
            print("ImageID reached maximum capacity ({max_images}). Resetting ID and changing state to 'ResetPending'.")
