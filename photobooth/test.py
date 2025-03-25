from nextdraw import NextDraw

class Plotter:
    def __init__(self):
        print("Starting Plotter ...")
        self.nd1 = NextDraw()
        self.nd1.interactive()  # Start interactive mode
        
        # Skip homing
        self.nd1.options.homing = False  # Make sure automatic homing is off

        if self.nd1.connect():
            print("Plotter connected successfully.")
        else:
            print("Failed to connect to the plotter.")
            quit()

    def plot(self):
        """Test simple plot movements."""
        self.nd1.moveto(1, 1)
        self.nd1.lineto(2, 1)
        self.nd1.moveto(0, 0)
        self.nd1.disconnect()

# Run the plotter
plotter = Plotter()
plotter.plot()
