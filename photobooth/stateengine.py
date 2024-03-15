from .display import Display

class StateEngine:
    def __init__(self):
        self.display = Display()

    def update_state(self, new_state):
        # Logic to update the current state
        self.display.update_status(new_state)