import os
import time
import LCD_1in44
from PIL import Image, ImageSequence
import paho.mqtt.client as mqtt

# LCD
LCD = LCD_1in44.LCD()
Lcd_ScanDir = LCD_1in44.SCAN_DIR_DFT

# MQTT
broker_address = "localhost"
client = mqtt.Client("LCD_Client", protocol=mqtt.MQTTv311)
client.connect(broker_address)

# Display images
# ------------------------------------------------------------------------
def display_default_image(LCD):
    default_image_path = 'assets/Waiting.jpg'
    try:
        image = Image.open(default_image_path)
        print("Displaying default image.")
        LCD.LCD_ShowImage(image, 0, 0)
    except Exception as e:
        print(f"Error displaying default image: {e}")
        
        
def display_image_based_on_state(LCD, state, rotation_angle=-90):
    state_to_image_path = {
        "Waiting": "assets/Waiting.jpg",
        "Tracking": "assets/Tracking.jpg",
        "Processing": "assets/Processing.jpg",
        "Drawing": "assets/Drawing.jpg",
        "ResetPending": "assets/Change.jpg",
        "Test": "assets/Change.jpg",
    }
    
    image_path = state_to_image_path.get(state)
    
    if image_path:
        try:
            image = Image.open(image_path)
            rotated_image = image.rotate(rotation_angle, expand=True)
            print(f"Displaying image for {state} state.")
            LCD.LCD_ShowImage(rotated_image, 0, 0)
        except Exception as e:
            print(f"Error displaying image for {state} state: {e}")
    else:
        print(f"No image mapping found for state: {state}")


# Messages
# ------------------------------------------------------------------------
def on_message(client, userdata, message):
    print(f"Received message on topic '{message.topic}': {message.payload.decode()}")

    # Check if the message is related to state changes
    if message.topic == "state_engine/state":
        state = message.payload.decode()
        print(f"{state}")
        # Call the function to display the image based on the received state
        display_image_based_on_state(LCD, state)

def publish_message(topic, message):
    client.publish(topic, message)


# Main
# ------------------------------------------------------------------------
def main():
    
    # LCD
    LCD.LCD_Init(Lcd_ScanDir)
    LCD.LCD_Clear()
    display_default_image(LCD)

    # MQTT
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
            if LCD.digital_read(LCD.GPIO_KEY1_PIN) != 0:
                publish_message("lcd/buttons", "KEY1")
            if LCD.digital_read(LCD.GPIO_KEY2_PIN) != 0:
                publish_message("lcd/buttons", "KEY2")
            if LCD.digital_read(LCD.GPIO_KEY3_PIN) != 0:
                publish_message("lcd/buttons", "KEY3")

            # Listen for MQTT messages
            client.loop()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")

if __name__ == '__main__':
    main()
