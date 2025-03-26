# nextdraw_conf.py
# Part of the NextDraw driver software
#
# Version 1.1.0, dated 2024-05-27.
#
# Copyright 2024 Windell H. Oskay, Bantam Tools

'''
Primary user-adjustable control parameters:

We encourage you to freely tune these values as needed to match the
 behavior and performance of your NextDraw to your application and taste.

These parameters are used as defaults when using NextDraw with the command-
 line interface (CLI) or with the python API. With the CLI, you can make
 copies of this configuration file and specify one as a configuration file.
 When using the Python API, override individual settings within your script.

 If you are operating the NextDraw from within Inkscape, please set your
  preferences within Inkscape. For settings that appear both here and in the
  Inkscape GUI, those in Inkscape will be used and those in this file will
  be ignored. Other settings can be configured here.

Settings within Inkscape only affect use within Inkscape, and do not affect
 the behavior of the NextDraw CLI or Python APIs.
'''

# DEFAULT VALUES

mode = 'plot'           # Operational mode or GUI tab. Default: 'plot'

handling = 1            # Handling mode (1-4). Choose your preferred option.
                            # 1: Technical drawing
                            # 2: Handwriting
                            # 3: Sketching
                            # 4: Constant speed

speed_pendown = 25      # Maximum plotting speed, when pen is down (1-100). Default 25
speed_penup = 75        # Maximum transit speed, when pen is up (1-100). Default 75
accel = 75              # Acceleration rate factor (1-100). Default 75

pen_pos_up = 60         # Height of pen when raised (0-100). Default 60
pen_pos_down = 40       # Height of pen when lowered (0-100). Default 40

pen_rate_raise = 75     # Rate of raising pen (1-100). Default 75
pen_rate_lower = 50     # Rate of lowering pen (1-100). Default 50

report_time = False     # Report time elapsed. Default False

default_layer = 1       # Layer(s) selected for layers mode (1-1000). Default 1

utility_cmd = 'read_name'   # Utility command to execute when in utility mode.
                                # Default 'read_name'

dist = 1.0              # Distance to walk in "walking" utility commands or changing
                            # resume position in the APIs. Variable units. Default 1.0

copies = 1              # Copies to plot, or 0 for continuous plotting. Default: 1
page_delay = 15         # Optional delay between copies (s). Default 15

preview = False         # Preview mode; simulate plotting only. Default False
rendering = True        # Enable rendering when running previews. Default True


model = 2               # Plotter hardware model (1-10). Set the model you use!
                            # 1: AxiDraw V2, V3, SE/A4. 2: AxiDraw V3/A3 or SE/A3.
                            # 3: AxiDraw V3 XLX. 4: AxiDraw MiniKit.
                            # 5: AxiDraw SE/A1.  6: AxiDraw SE/A2.
                            # 7: AxiDraw V3/B6.
                            # 8: Bantam Tools NextDraw™ 8511. (Default)
                            # 9: Bantam Tools NextDraw™ 1117.
                            # 10: Bantam Tools NextDraw™ 2234.

penlift = 3             # pen lift servo configuration (1-3).
                            # 1 or 2: Default for plotter model
                            # 3: Narrow-band brushless servo

port = None             # Serial port or named plotter to use
                            # None (Default) will plot to first unit located

port_config = 0         # Serial port behavior option (0-2)
                            # 0: Plot to first unit found, unless port is specified (Default)
                            # 1: Plot to first machine located
                            # 2: Plot to a specific machine only, given by port

homing = True           # If True (Default), enable Automatic homing on supported models

auto_rotate = True      # Auto-select portrait vs landscape orientation
                            # Default: True

reordering = 2          # Plot optimization option (0-4; 3 is deprecated)
                            # 0: Least; Only connect adjoining paths (Default)
                            # 1: Basic; Also reorder paths for speed
                            # 2: Full; Also allow path reversal
                            # 4: None; Strictly preserve file order

random_start = False    # Randomize start locations of closed paths. Default False

hiding = False          # Hidden-line removal. Default: False

webhook = False         # Enable webhook alerts when True
                            # Default False

webhook_url = None      # URL for webhook alerts. Default None

digest = 0              # Plot digest output option. (NOT supported in Inkscape context.)
                            # 0: Disabled; No change to behavior or output (Default)
                            # 1: Output "plob" digest, not full SVG, when saving file
                            # 2: Disable plots and previews; generate digest only

progress = False        # Enable progress bar display in NextDraw CLI, when True
                            # Default False
                            # This option has no effect in Inkscape or Python API contexts.

'''
Additional user-adjustable control parameters.
Values below this point are configured only in this file, not through the user interface(s).
'''

pause_warning = True    # If True (default), cancel and give a warning, _once_ if you try to
                        #   start a paused plot from the beginning instead of resuming. You can
                        #   cancel a single paused plot with the "strip data" utility command.

servo_timeout = 60000   # Time, ms, for servo motor to power down after last movement command
                        #   (default: 60000). This feature requires EBB v 2.5 hardware (with USB
                        #   micro not USB mini connector), servo_pin set to 1 (only).
                        #   Not applicable for use with brushless pen-lift motor.

check_updates = True    # If True, allow NextDraw software to check online to see
                        #    what the current software version is, when you
                        #    query the version. Set to False to disable.

use_b3_out = False      # If True, enable digital output pin B3, which will be high (3.3V)
                        #   when the pen is down, and low otherwise. Can be used to control
                        #   external devices like valves, relays, or lasers.

auto_rotate_ccw = True  # If True (default), auto-rotate is counter-clockwise when active.
                        #   If False, auto-rotate direction is clockwise.

options_message = True  # If True (default), display an advisory message if Apply is clicked
                        #   in the Inkscape GUI, while in tabs that have no effect.
                        #   (Clicking Apply on these tabs has no effect other than the message.)
                        #   This message can prevent the situation where one clicks Apply on the
                        #   Options tab and then waits a few minutes before realizing that
                        #   no plot has been initiated.

report_lifts = False    # Report number of pen lifts when reporting plot duration (Default: False)

auto_clip_lift = True   # Option applicable only to XY movements in the Interactive Python API.
                        #   If True (default), keep pen up when motion is clipped by travel bounds.

preview_paths = 3       # Preview mode rendering option (0-3):
                            # 0: Do not render previews
                            # 1: Render only pen-down movement
                            # 2: Render only pen-up movement
                            # 3: Render all movement (Default)

# Colors used to represent pen-up and pen-down travel in preview mode:

preview_color_up = 'LightPink' # Pen-up travel color. Default: LightPink; rgb(255, 182, 193)
preview_color_down = 'Blue'    # Pen-up travel color. Default: Blue; rgb(0, 0, 255)

skip_voltage_check = False  # Set True to disable EBB input power voltage checks. Default: False
                            # Some functions, including automatic homing, require voltage checks

clip_to_page = True  # Clip plotting area to SVG document size. Default: True

min_gap = 0.006     # Automatic path joining threshold, inches. Default: 0.006
                    # If greater than zero, pen-up moves shorter than this distance
                    #   will be replaced by pen-down moves. Set negative to disable.
                    # Setting reordering to 4 (strict) will also disable path joining.

pen_delay_up = 0        # Optional delay after pen is raised (ms). Default 0
pen_delay_down = 0      # Optional delay after pen is lowered (ms). Default 0


'''
Secondary control parameters:

Values below this point are configured here, not through the other user interfaces.
These values are carefully chosen, and generally do not need to be adjusted in everyday use.
Be aware that one can easily change these values such that things will not work properly,
or at least not how you expect them to. Edit with caution, and keep a backup copy.
'''


'''
Parameter overrides

Certain machine configuration parameters, including the model name, travel, and type of
pen-lift motor, are normally set automatically based on the model and handling options.
Similarly, the configuration parameters for the pen-lift servo are set based which type
of pen-lift motor is in use, which is determined by the "model" option, in combination
with the "penlift" option.

This "overrides" dictionary provides a mechanism of overriding one or more of those values,
after the software processes and applies the defaults applied by "model", "handling",
and "penlift". If one or more parameters in the overrides dictionary have a value other
than None, that value is applied to the given parameter *after* the other values are
applied.
'''

overrides = {
    'model_name': None,         # Human-readable plotter model name
    'travel_x': 16.93,          # x-travel, inches
    'travel_y': 11.69,          # y-travel, inches
    'jerk_pen_up': None,        # maximum pen-up jerk value in/s^3
    'jerk_pen_down': None,      # maximum pen-down jerk value in/s^3
    'speed_limit':None,         # Speed limit, inches per second.
    'speed_up':None,            # Speed limit, pen-up, inches per second.
    'auto_home': None,          # Boolean; True if model supports automatic homing
    'z_motor': None,            # Machine default is 0 for standard servo, 1 for brushless
    'servo_pin': None,          # EBB I/O pin number (port B) for pen-lift servo
    'servo_max': None,          # Up, "100%" position. Units of 83.3 ns
    'servo_min': None,          # Down, "0%" position. Units of 83.3 ns
    'servo_sweep_time': None,   # Time, ms, for servo to move over full range
    'servo_move_min': None,     # Minimum time, ms, for pen-lift movement
    'servo_move_slope': None,   # Added time, ms, per % of vertical travel.
    'resolution': None,         # Resolution 1: High (2874 steps/in), 2: Low (1437 steps/in)
    'curve_tolerance':None,     # Allowed deviation, inch, from original path data
    'const_speed':None,         # Use constant velocity mode when pen is down.
}


''' Additional Secondary control parameters: '''

native_res_factor = 1016.0  # Motor resolution factor, steps per inch. Default: 1016.0
# Note that resolution is defined along native (not X or Y) axes.
# Resolution is native_res_factor * sqrt(2) steps/inch in Low Resolution  (Approx 1437 steps/in)
#       and 2 * native_res_factor * sqrt(2) steps/inch in High Resolution (Approx 2874 steps/in)

max_step_rate = 24.995  # Maximum allowed motor step rate, in steps per millisecond.
# The absolute maximum step rate for the EBB is 25 kHz. Faster movements may result in a crash
# (loss of position control). This value is normally used _for speed limit checking only_.

speed_lim_xy_lr = 17.3958 # Max XY speed allowed in Low Resolution mode, in/s. Max: 17.3958
speed_lim_xy_hr = 8.6979  # Max XY speed allowed in High Resolution mode, in/s. Max: 8.6979
# Do not increase these values above Max; they are derived from max_step_rate and the resolution.

button_interval = 0.05  # Minimum interval (s), for polling pause button. Default: 0.05 (50 ms)

bounds_tolerance = 0.003  # Suppress warnings if bounds are exceeded by less than this distance (inches).
