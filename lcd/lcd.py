import LCD_1in44
import time
from PIL import Image

# Function to display an image on the LCD
def display_image_on_lcd(image_path, LCD):
    try:
        image = Image.open(image_path)
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying {image_path}: {e}")
        # Display default image in case of an error
        display_default_image(LCD)

# Function to display a default image
def display_default_image(LCD):
    default_image_path = '../assets/default.jpg'  # Adjusted path
    try:
        image = Image.open(default_image_path)
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying default image: {e}")

def main():
    LCD = LCD_1in44.LCD()
    print("Starting LCD...")
    Lcd_ScanDir = LCD_1in44.SCAN_DIR_DFT  # SCAN_DIR_DFT = D2U_L2R
    LCD.LCD_Init(Lcd_ScanDir)
    LCD.LCD_Clear()

    # Set the path to the FIFO
    fifo_path = '/tmp/state_fifo'

    # Initialize display with a default image
    display_default_image(LCD)

    # Read from FIFO and update the LCD with new images
    try:
        with open(fifo_path, 'r') as fifo:
            while True:
                image_path = fifo.readline().strip()
                if image_path:
                    print(f"Received request to display: {image_path}")
                    display_image_on_lcd(image_path, LCD)
                else:
                    # If the path is empty, continue displaying the default image
                    display_default_image(LCD)
    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")

if __name__ == '__main__':
    main()
