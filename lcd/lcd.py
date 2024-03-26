import os
import time
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

    display_fifo_path = 'lcd/tmp/state_fifo'
    key_fifo_path = 'lcd/tmp/key_fifo'
    assets_dir = 'assets/'

    # Create FIFOs if they don't exist
    if not os.path.exists(display_fifo_path):
        os.mkfifo(display_fifo_path)
    if not os.path.exists(key_fifo_path):
        os.mkfifo(key_fifo_path)

    display_default_image(LCD)

    try:
        with open(display_fifo_path, 'r') as fifo, open(key_fifo_path, 'w') as key_fifo:
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

                # Check for key presses and write to key_fifo
                if LCD.digital_read(LCD.GPIO_KEY_UP_PIN) == 0:
                    key_fifo.write("Up\n")
                if LCD.digital_read(LCD.GPIO_KEY_DOWN_PIN) == 0:
                    key_fifo.write("Down\n")
                if LCD.digital_read(LCD.GPIO_KEY_LEFT_PIN) == 0:
                    key_fifo.write("Left\n")
                if LCD.digital_read(LCD.GPIO_KEY_RIGHT_PIN) == 0:
                    key_fifo.write("Right\n")
                if LCD.digital_read(LCD.GPIO_KEY_PRESS_PIN) == 0:
                    print(f"Pressed")
                    key_fifo.write("Press\n")
                if LCD.digital_read(LCD.GPIO_KEY1_PIN) == 0:
                    key_fifo.write("KEY1\n")
                if LCD.digital_read(LCD.GPIO_KEY2_PIN) == 0:
                    key_fifo.write("KEY2\n")
                if LCD.digital_read(LCD.GPIO_KEY3_PIN) == 0:
                    key_fifo.write("KEY3\n")

                # Flush the FIFO to ensure the message is sent
                key_fifo.flush()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")

if __name__ == '__main__':
    main()
