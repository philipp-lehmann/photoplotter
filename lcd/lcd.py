import time
import LCD_1in44
from PIL import Image
import paho.mqtt.client as mqtt
import threading
import psutil  # Import psutil to check for running processes
import subprocess  # Import subprocess to start main.py

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

# Threading management
current_thread = None
stop_event = threading.Event()

# Function to check if main.py is running
def is_main_py_running():
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'main.py' in process.info['cmdline']:
            return True
    return False

# Function to start main.py if not running
def start_main_py():
    # Path to your virtual environment's Python interpreter and main.py script
    venv_python = "photoplotter-env/bin/python"
    main_script = "main.py"
    subprocess.Popen([venv_python, main_script])


    
# Display images
# ------------------------------------------------------------------------
def display_default_image(LCD):
    default_image_path = 'assets/display/Ready.jpg'
    try:
        image = Image.open(default_image_path)
        rotated_image = image.rotate(-90, expand=True)
        print("Displaying default image.")
        LCD.LCD_ShowImage(rotated_image, 0, 0)
    except Exception as e:
        print(f"Error displaying default image: {e}")
        
# Function to display "Processing-9.jpg" when main.py is not running
def display_error_image(LCD):
    print("Displaying Processing-9.jpg as main.py is not running.")
    state = "Processing"
    display_image_based_on_state(LCD, state)  # This will show the 'Processing' state, which includes "Processing-9.jpg"

def display_image_series(LCD, image_paths, display_time=1, loop=False):
    start_time = time.time()
    current_image_index = 0
    total_images = len(image_paths)
    
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= display_time:
            start_time = time.time()
            
            try:
                image_path = image_paths[current_image_index]
                image = Image.open(image_path)
                rotated_image = image.rotate(-90, expand=True)
                LCD.LCD_ShowImage(rotated_image, 0, 0)
            except Exception as e:
                print(f"Error displaying image: {e}")
            
            current_image_index += 1
            
            if current_image_index >= total_images:
                if loop:
                    current_image_index = 0
                else:
                    break  # Stop if not looping
        
        time.sleep(0.001)

def display_image_based_on_state(LCD, state):
    global current_thread, stop_event

    state_config = {
        "Waiting": {
            "images": ["assets/display/Waiting-0.jpg", "assets/display/Waiting-1.jpg", "assets/display/Waiting-2.jpg", "assets/display/Waiting-3.jpg", "assets/display/Waiting-4.jpg"],
            "display_time": 0.125,
            "loop": True
        },
        "Working": {
            "images": ["assets/display/Working-0.jpg", "assets/display/Working-1.jpg", "assets/display/Working-2.jpg", "assets/display/Working-3.jpg", "assets/display/Working-4.jpg", "assets/display/Working-5.jpg", "assets/display/Working-6.jpg", "assets/display/Working-7.jpg", "assets/display/Working-8.jpg", "assets/display/Working-9.jpg"],
            "display_time": 0.125,
            "loop": True
        },
        "Tracking": {
            "images": ["assets/display/Tracking-0.jpg", "assets/display/Tracking-1.jpg", "assets/display/Tracking-2.jpg", "assets/display/Tracking-3.jpg", "assets/display/Tracking-4.jpg", "assets/display/Tracking-5.jpg", "assets/display/Tracking-6.jpg", "assets/display/Tracking-7.jpg", "assets/display/Tracking-8.jpg", "assets/display/Tracking-9.jpg"],
            "display_time": 0.125,
            "loop": True
        },
        "Snapping": {
            "images": ["assets/display/Snapping-0.jpg", "assets/display/Snapping-1.jpg", "assets/display/Snapping-2.jpg", "assets/display/Snapping-3.jpg", "assets/display/Snapping-4.jpg", "assets/display/Snapping-5.jpg", "assets/display/Snapping-6.jpg", "assets/display/Snapping-7.jpg", "assets/display/Snapping-8.jpg", "assets/display/Snapping-9.jpg"],
            "display_time": 0.125,
            "loop": True
        },
        "Processing": {
            "images": ["assets/display/Processing-1.jpg", "assets/display/Processing-2.jpg", "assets/display/Processing-3.jpg", "assets/display/Processing-4.jpg", "assets/display/Processing-5.jpg", "assets/display/Processing-6.jpg", "assets/display/Processing-7.jpg", "assets/display/Processing-8.jpg", "assets/display/Processing-9.jpg", "assets/display/Processing-10.jpg"],
            "display_time": 0.125,
            "loop": True
        },
        "ResetPending": {
            "images": ["assets/display/ResetPending.jpg"],
            "display_time": 1,
            "loop": False
        },
        "Test": {
            "images": ["assets/display/Test.jpg"],
            "display_time": 1,
            "loop": False
        },
    }

    num_images = 15
    for i in range(1, num_images + 1):
        state_config[f"Drawing-{i}"] = {
            "images": [f"assets/display/Drawing-{i}.jpg"],
            "display_time": 0.1,
            "loop": False
        }

    config = state_config.get(state)
    if config:
        # Signal the current thread to stop
        if current_thread and current_thread.is_alive():
            stop_event.set()
            current_thread.join()  # Wait for the thread to finish

        # Clear the stop event and start a new thread
        stop_event.clear()
        current_thread = threading.Thread(target=display_image_series, args=(LCD, config['images'], config['display_time'], config['loop']))
        current_thread.start()
    else:
        print(f"No configuration found for state: {state}")

# Messages
# ------------------------------------------------------------------------
def on_message(client, userdata, message):
    # print(f"Received message on topic '{message.topic}': {message.payload.decode()}")
    if message.topic == "state_engine/state":
        state = message.payload.decode()
        print(f"{state}")
        display_image_based_on_state(LCD, state)

def publish_message(topic, message):
    client.publish(topic, message)

# Button press check and trigger main.py restart
def check_button(LCD, button_pin, button_name):
    current_time = time.time()
    if LCD.digital_read(button_pin) == 1:
        if (current_time - last_press_time[button_name]) > debounce_delay:
            last_press_time[button_name] = current_time
            if button_name == "KEY1":  # Use KEY1 to check and start main.py
                if not is_main_py_running():
                    print("main.py is not running.")
                    display_error_image(LCD)  # Display "Processing-9.jpg"
                    start_main_py()  # Start main.py
                else:
                    print("main.py is already running.")
            return True
    return False

# Main logic
def main():
    LCD.LCD_Init(Lcd_ScanDir)
    LCD.LCD_Clear()
    display_default_image(LCD)

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

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nExiting LCD due to keyboard interrupt...")
        client.loop_stop()

if __name__ == '__main__':
    main()
