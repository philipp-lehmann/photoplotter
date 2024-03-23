import os
import LCD_1in44
from PIL import Image, ImageSequence

def display_default_image(LCD):
    default_image_path = 'assets/Waiting.jpg'
    try:
        image = Image.open(default_image_path)
        print("Displaying default image.")
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying default image: {e}")
        
        
def display_image_on_lcd(image_path, LCD):
    try:
        image = Image.open(image_path)
        print(f"Displaying {image_path}")
        LCD.LCD_Clear()
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying {image_path}: {e}")
        display_default_image(LCD)


def display_gif_on_lcd(gif_path, LCD, duration_per_frame=0.1):
    try:
        # Open the gif file
        with Image.open(gif_path) as img:
            for frame in ImageSequence.Iterator(img):
                frame = frame.convert("RGB")
                frame = frame.resize((LCD.LCD_Dis_Column, LCD.LCD_Dis_Page))
                LCD.LCD_ShowImage(frame, 0, 0)
                time.sleep(duration_per_frame)
    except Exception as e:
        print(f"Failed to display GIF: {e}")
        display_default_image(LCD)
        
        
def main():
    LCD = LCD_1in44.LCD()
    Lcd_ScanDir = LCD_1in44.SCAN_DIR_DFT
    LCD.LCD_Init(Lcd_ScanDir)
    LCD.LCD_Clear()

    fifo_path = 'lcd/tmp/state_fifo'
    assets_dir = 'assets/'

    if not os.path.exists(fifo_path):
        os.mkfifo(fifo_path)

    display_default_image(LCD)

    try:
        with open(fifo_path, 'r') as fifo:
            while True:
                filename = fifo.readline().strip()
                if filename:
                    image_path = os.path.join(assets_dir, filename)
                    if os.path.exists(image_path):
                        print(f"Received request to display: {image_path}")
                        display_image_on_lcd(image_path, LCD)
                    else:
                        print(f"File not found: {image_path}, displaying default image.")
                        display_default_image(LCD)
                else:
                    print("Received empty filename, keeping current image.")
                    # Optionally, re-display the current image or do nothing.
    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")

if __name__ == '__main__':
    main()
