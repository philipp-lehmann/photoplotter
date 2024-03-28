import os
import time
import LCD_1in44
from PIL import Image, ImageSequence
import paho.mqtt.client as mqtt

broker_address = "localhost"
client = mqtt.Client("LCD_Client", protocol=mqtt.MQTTv311)
client.connect(broker_address)  # Connect to the broker

def publish_message(topic, message):
    client.publish(topic, message)

def display_default_image(LCD):
    default_image_path = 'assets/Waiting.jpg'
    try:
        image = Image.open(default_image_path)
        print("Displaying default image.")
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying default image: {e}")

def on_message(client, userdata, message):
    # Handle messages received on subscribed topics
    print(f"Received message on topic '{message.topic}': {message.payload.decode()}")
    # Add logic to handle incoming messages as needed
    # Example: Update LCD display based on the received message

def main():
    LCD = LCD_1in44.LCD()
    Lcd_ScanDir = LCD_1in44.SCAN_DIR_DFT
    LCD.LCD_Init(Lcd_ScanDir)
    LCD.LCD_Clear()

    assets_dir = 'assets/'
    display_default_image(LCD)

    # Subscribe to messages from the StateEngine
    client.subscribe("state_engine/state")
    client.on_message = on_message

    try:
        while True:
            # Check for key presses and write to key_fifo
            if LCD.digital_read(LCD.GPIO_KEY_UP_PIN) != 0:
                publish_message("lcd/buttons", "UP")
            if LCD.digital_read(LCD.GPIO_KEY_DOWN_PIN) != 0:
                publish_message("lcd/buttons", "DOWN")
            if LCD.digital_read(LCD.GPIO_KEY_LEFT_PIN) != 0:
                publish_message("lcd/buttons", "LEFT")
            if LCD.digital_read(LCD.GPIO_KEY_RIGHT_PIN) != 0:
                publish_message("lcd/buttons", "RIGHT")
            if LCD.digital_read(LCD.GPIO_KEY_PRESS_PIN) != 0:
                publish_message("lcd/buttons", "PRESS")
                print(f"press")
            if LCD.digital_read(LCD.GPIO_KEY1_PIN) != 0:
                publish_message("lcd/buttons", "KEY1")
            if LCD.digital_read(LCD.GPIO_KEY2_PIN) != 0:
                publish_message("lcd/buttons", "KEY2")
            if LCD.digital_read(LCD.GPIO_KEY3_PIN) != 0:
                publish_message("lcd/buttons", "KEY3")

            # Listen for MQTT messages
            client.loop()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")

if __name__ == '__main__':
    main()
