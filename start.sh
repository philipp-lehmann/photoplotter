#!/bin/bash

# Path to the LCD script within the photoplotter directory
LCD_SCRIPT_PATH="lcd/lcd.py"

# Starting the LCD script with sudo but without activating a virtual environment
echo "Starting LCD script with sudo..."
sudo python $LCD_SCRIPT_PATH &
LCD_PID=$!

# Ensure to give a moment for the LCD script to initialize (optional)
sleep 2

# Activate the virtual environment for the main application
echo "Activating virtual environment for main application..."
source photoplotter-env/bin/activate

# Run the main application
echo "Running main application..."
python main.py

# Cleanup
echo "Main application finished. Cleaning up..."
sudo kill $LCD_PID
echo "LCD script stopped."

# Deactivate the virtual environment (if the script is run in a new shell)
deactivate
