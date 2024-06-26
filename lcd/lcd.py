import time
import LCD_1in44
from PIL import Image
import paho.mqtt.client as mqtt

# LCD
LCD = LCD_1in44.LCD()
Lcd_ScanDir = LCD_1in44.SCAN_DIR_DFT
last_press_time = {
    "UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0, "PRESS": 0,
    "KEY1": 0, "KEY2": 0, "KEY3": 0
}
debounce_delay = 0.02


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
        rotated_image = image.rotate(-90, expand=True) 
        print("Displaying default image.")
        LCD.LCD_ShowImage(rotated_image, 0, 0)
    except Exception as e:
        print(f"Error displaying default image: {e}")

def display_image_series(LCD, image_paths, display_time=1, loop=False):
    start_time = time.time()
    current_image_index = 0
    total_images = len(image_paths)
    
    # Initially set to False, will be updated based on loop and operation
    completed = False 
    
    while not completed:
        elapsed_time = time.time() - start_time
        
        # Check if it's time to switch to the next image
        if elapsed_time >= display_time:
            start_time = time.time()  # Reset the start time
            
            # Load and display the current image
            try:
                image_path = image_paths[current_image_index]
                image = Image.open(image_path)
                rotated_image = image.rotate(-90, expand=True) 
                LCD.LCD_ShowImage(rotated_image, 0, 0)
            except Exception as e:
                print(f"Error displaying image: {e}")
            
            # Move to the next image
            current_image_index += 1
            
            # Check if we've displayed all images
            if current_image_index >= total_images:
                if loop:
                    current_image_index = 0  # Reset to the first image for looping
                else:
                    completed = True  # Stop if not looping
        
        # Non-blocking wait (very short sleep to reduce CPU usage)
        time.sleep(0.001) 


def display_image_based_on_state(LCD, state):
    # Configuration for each state with paths, timings, and loop settings
    state_config = {
        "Waiting": {
            "images": ["assets/Ready-3.jpg", "assets/Ready-2.jpg", "assets/Ready-1.jpg"],
            "display_time": 0.5,
            "loop": False
        },
        "Tracking": {
            "images": ["assets/Tracking-4.jpg", "assets/Tracking-3.jpg", "assets/Tracking-2.jpg", "assets/Tracking-1.jpg"],
            "display_time": 1,
            "loop": False
        },
        "Processing": {
            "images": ["assets/Processing-1.jpg", "assets/Processing-2.jpg", "assets/Processing-3.jpg", "assets/Processing-4.jpg"],
            "display_time": 0.1,
            "loop": False
        },
        "Drawing": {
            "images": ["assets/Drawing-04.jpg", "assets/Drawing-03.jpg", "assets/Drawing-02.jpg", "assets/Drawing-01.jpg"],
            "display_time": 0.1,
            "loop": False
        },
        "ResetPending": {
            "images": ["assets/ResetPending.jpg"],
            "display_time": 1,
            "loop": False
        },
        "Test": {
            "images": ["assets/Test.jpg"],
            "display_time": 1,
            "loop": False
        },
    }
    
    # Define the number of images
    num_images = 15

    # Loop through the range and create a unique key for each "Drawing" state
    for i in range(1, num_images + 1):
        state_config[f"Drawing-{i:02}"] = {
            "images": [f"assets/Drawing-{i:02}.jpg"],
            "display_time": 0.1,
            "loop": False
        }

    state_config.update(state_config)  
    config = state_config.get(state)
    if config:
        display_image_series(LCD, config['images'], config['display_time'], config['loop'])
    else:
        print(f"No configuration found for state: {state}")      


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


def check_button(LCD, button_pin, button_name):
    current_time = time.time()
    if LCD.digital_read(button_pin) == 1:
        if (current_time - last_press_time[button_name]) > debounce_delay:
            last_press_time[button_name] = current_time
            return True
    return False

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
    client.loop_start()
    
    try:
        while True:
            
            if check_button(LCD, LCD.GPIO_KEY_UP_PIN, "UP"):
                publish_message("lcd/buttons", "UP")
            if check_button(LCD, LCD.GPIO_KEY_DOWN_PIN, "DOWN"):
                publish_message("lcd/buttons", "DOWN")
            if check_button(LCD, LCD.GPIO_KEY_LEFT_PIN, "LEFT"):
                publish_message("lcd/buttons", "LEFT")
            if check_button(LCD, LCD.GPIO_KEY_RIGHT_PIN, "RIGHT"):
                publish_message("lcd/buttons", "RIGHT")
            if check_button(LCD, LCD.GPIO_KEY_PRESS_PIN, "PRESS"):
                publish_message("lcd/buttons", "PRESS")
            if check_button(LCD, LCD.GPIO_KEY1_PIN, "KEY1"):
                publish_message("lcd/buttons", "KEY1")
            if check_button(LCD, LCD.GPIO_KEY2_PIN, "KEY2"):
                publish_message("lcd/buttons", "KEY2")
            if check_button(LCD, LCD.GPIO_KEY3_PIN, "KEY3"):
                publish_message("lcd/buttons", "KEY3")

            # Listen for MQTT messages
            # client.loop()
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")
        client.loop_stop() 

if __name__ == '__main__':
    main()
