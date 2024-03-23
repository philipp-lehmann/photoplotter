import os
import LCD_1in44
from PIL import Image

# Function to display an image on the LCD
def display_image_on_lcd(image_path, LCD):
    try:
        image = Image.open(image_path)
        print(f"Displaying {image_path}")
        LCD.LCD_Clear()
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying {image_path}: {e}")
        # Display default image in case of an error
        display_default_image(LCD)

# Function to display a default image
def display_default_image(LCD):
    # Assuming your script runs from the photoplotter directory
    default_image_path = '../assets/Ready.jpg'
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

    # Set paths
    fifo_path = 'tmp/state_fifo'
    assets_dir = '../assets/'

    # Ensure the FIFO exists before attempting to read from it
    if not os.path.exists(fifo_path):
        os.mkfifo(fifo_path)

    # Initialize display with a default image
    display_default_image(LCD)

    # Continuously read from FIFO and update the LCD with new images
    try:
        with open(fifo_path, 'r') as fifo:
            while True:
                filename = fifo.readline().strip()  # Filename from FIFO
                if filename:
                    image_path = os.path.join(assets_dir, filename)
                    print(f"Received request to display: {image_path}")
                    display_image_on_lcd(image_path, LCD)
                else:
                    # If the path is empty, this will ensure the default image is displayed
                    display_default_image(LCD)
    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")

if __name__ == '__main__':
    main()
