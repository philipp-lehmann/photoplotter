import os
from nextdraw import NextDraw

class Plotter:
    def __init__(self):
        print("Starting Plotter ...")
        self.nd1 = NextDraw()
        self.nd1.interactive() 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "nextdraw_conf.py")
        self.nd1.load_config(config_path)
 
        self.plotter_found = self.connect_to_plotter()

        if self.nd1.connect():
            print("Plotter connected successfully.")
            print(f"NextDraw connected. Model: {self.nd1.options.model}" )

        else:
            print("Failed to connect to the plotter.")
            
        self.nd1.plot_setup()
        self.nd1.plot_run()

    # def plot(self):
    #     """Test simple plot movements."""
    #     self.nd1.moveto(1, 1)
    #     self.nd1.lineto(2, 1)
    #     self.nd1.moveto(0, 0)
    #     self.nd1.disconnect()
        
    def connect_to_plotter(self):
        """Attempt to connect to the NextDraw plotter."""
        if self.nd1.connect():
            self.nd1.options.model = 2
            self.nd1.options.auto_rotate = True
            self.nd1.options.pen_rate_lower = 80
            self.nd1.options.pen_rate_raise = 80
            self.nd1.options.speed_pendown = 100
            self.nd1.options.speed_penup = 100
            self.nd1.options.penlift = 3
            
            self.nd1.update()
            print(f"NextDraw connected. Model: {self.nd1.options.model}")
            return True
        else:
            print("NextDraw not found. Entering simulation mode.")
            return False

    def return_home(self):
        if self.plotter_found:
            # New code to return home
            print("Plotter: Returning home.")
        else:
            print("Simulation Mode: Returning home.")

    def plot_image(self, svg_path):
           
        svg_path = os.path.abspath(svg_path)
        
        if os.path.exists(svg_path):
            if self.plotter_found:
                self.nd1.interactive()   
                self.nd1.plot_setup(svg_path)
                self.nd1.plot_run(True)
                
                print("Plotting complete.")
            else:
                print("NextDraw not found.")
        else:
            print("SVG file not found.")
        
        
