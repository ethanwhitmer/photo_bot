from serial import Serial # pip install pyserial
import time
import redis
from pynput import keyboard

# Connect to the Arduino (replace first argument with correct port)
arduino = Serial('/dev/tty.usbmodem1101', 9600)

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

global state
state = 'open'

def run_motor(direction, duration):
    if direction == 'left':
        arduino.write(b'L')
    elif direction == 'right':
        arduino.write(b'R')
    time.sleep(duration)
    arduino.write(b'S')

def on_press(key):
    global state
    duration = 3.25 # tune for pen size (3.25 works well for full open/close)
    try:
        if key == keyboard.Key.down: # close fully
            if state == 'open':
                state = 'closed'
                run_motor('left', duration)
        elif key == keyboard.Key.up: # open fully
            if state == 'closed':
                state = 'open'
                run_motor('right', duration)
        elif key == keyboard.Key.right: # open
            arduino.write(b'R')
        elif key == keyboard.Key.left: # close
            arduino.write(b'L')

    except AttributeError:
        pass  # Handle special keys that don't have a `char` attribute

def on_release(key):
    if key == keyboard.Key.left or key == keyboard.Key.right:
        arduino.write(b'S')
    if key == keyboard.Key.esc:
        # Stop listener
        return False

try:
    # Start the listener in a non-blocking way
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        # sent heartbeat signal
        arduino.write(b'H')
        time.sleep(0.5)


except KeyboardInterrupt:
    print("Exiting program")
    arduino.write(b'S')

finally:
    arduino.close()
