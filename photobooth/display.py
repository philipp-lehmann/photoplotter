# from .LCD_1in44 import LCD  # Import the LCD class from the LCD_1in44 module
# from PIL import Image, ImageDraw, ImageFont

class Display:
    def __init__(self):
        print("Starting Display ...")
        # Initialize the LCD display
        # self.LCD = LCD()  # Use the imported LCD class directly
        # Lcd_ScanDir = LCD.SCAN_DIR_DFT  # Use the SCAN_DIR_DFT from the LCD class
        # self.LCD.LCD_Init(Lcd_ScanDir)
        # self.LCD.LCD_Clear()

        # self.width = self.LCD.width
        # self.height = self.LCD.height

        # Optional: Load a default font, or specify your own path to a .ttf font file
        # self.font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 16)

    def update_status(self, status):
        # Create a new image with white background
        # image = Image.new("RGB", (self.width, self.height), "WHITE")
        # draw = ImageDraw.Draw(image)

        # # Draw a simple rectangle as a background for the text
        # # Modify or remove this as per your needs
        # draw.rectangle([(10, 10), (self.width - 10, 30)], fill="LIGHTGREY")

        # # Draw the status text
        # # Adjust the coordinates as needed to center or move your text
        # draw.text((15, 15), status, fill="BLACK")  # Use `font=self.font` if you've loaded a font

        # # Display the image on the LCD
        # self.LCD.LCD_ShowImage(image, 0, 0)

        print(f"Displaying status: {status}")
