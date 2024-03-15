from .display import Display

class StateEngine:
    def __init__(self):
        self.display = Display()
        self.state = "Startup"
        self.transitions = {
            "Startup": ["Ready"],
            "Ready": ["Tracking", "Processing"],
            "Tracking": ["Processing"],
            "Processing": ["Drawing", "Ready"],
            "Drawing": ["Ready"]
        }

    def change_state(self, new_state):
        if new_state in self.transitions[self.state]:
            print(f"Transitioning from {self.state} to {new_state}.")
            self.state = new_state
        else:
            print(f"Invalid transition from {self.state} to {new_state}.")

    def get_state(self):
        return self.state
    

    def update_state(self, new_state):
        # Logic to update the current state
        self.display.update_status(new_state)
    
    def change_state(self, new_state):
        self.current_state = new_state
        # Add any additional logic you need when changing states
        print(f"State changed to: {self.current_state}")