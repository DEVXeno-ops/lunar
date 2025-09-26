# launcher.py  -- Safe launcher for the demo detection (no mouse automation / no clicks)
import json
import os
import sys
import threading
import time

from pynput import keyboard
from termcolor import colored

# Import the safe demo class. If you named the demo file differently, update the import.
from lib.aimbot import AimbotDemo as Aimbot

# global instance set in main()
lunar = None

def on_release(key):
    """
    Keyboard listener callback (global toggle and exit).
    F1 toggles detection, F2 quits cleanly.
    """
    global lunar
    try:
        if key == keyboard.Key.f1:
            if lunar is not None:
                lunar.enabled = not lunar.enabled
                status = "ENABLED" if lunar.enabled else "DISABLED"
                print(f"[INFO] Detection {status}")
        elif key == keyboard.Key.f2:
            if lunar is not None:
                print("[INFO] F2 pressed. Requesting clean exit...")
                # Call the demo's safe_exit helper if present
                try:
                    lunar._safe_exit()
                except Exception:
                    # fallback to sys.exit
                    print("[INFO] Exiting now.")
                    os._exit(0)
    except Exception:
        # swallow exceptions from listener to avoid killing the thread
        pass

def setup():
    """
    Ask the user to input in-game sensitivities and save to lib/config/config.json.
    This matches the safe-demo's expected config keys.
    """
    path = "lib/config"
    if not os.path.exists(path):
        os.makedirs(path)

    print("[INFO] In-game X and Y axis sensitivity should be the same")
    def prompt(msg):
        valid_input = False
        while not valid_input:
            try:
                number = float(input(msg))
                valid_input = True
            except ValueError:
                print("[!] Invalid Input. Make sure to enter only the number (e.g. 6.9)")
        return number

    xy_sens = prompt("X-Axis and Y-Axis Sensitivity (from in-game settings): ")
    targeting_sens = prompt("Targeting Sensitivity (from in-game settings): ")

    print("[INFO] Your in-game targeting sensitivity must be the same as your scoping sensitivity")
    sensitivity_settings = {
        "xy_sens": xy_sens,
        "targeting_sens": targeting_sens,
        "xy_scale": 10 / xy_sens,
        "targeting_scale": 1000 / (targeting_sens * xy_sens)
    }

    with open('lib/config/config.json', 'w') as outfile:
        json.dump(sensitivity_settings, outfile, indent=2)
    print("[INFO] Sensitivity configuration complete")

def main():
    global lunar
    lunar = Aimbot(collect_data = ("collect_data" in sys.argv), box_constant = 416, debug = False)
    # start the demo in the main thread (it handles its own loop and cv2 window)
    try:
        lunar.start()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt caught, exiting...")
        try:
            lunar._safe_exit()
        except Exception:
            os._exit(0)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

    print(colored('''
    | |
    | |    _   _ _ __   __ _ _ __
    | |   | | | | '_ \\ / _` | '__|
    | |___| |_| | | | | (_| | |
    \\_____/\__,_|_| |_|\\__,_|_|

    (Neural Network Detection Demo — SAFE)''', "yellow"))

    path_exists = os.path.exists("lib/config/config.json")
    if not path_exists or ("setup" in sys.argv):
        if not path_exists:
            print("[!] Sensitivity configuration is not set")
        setup()

    # ensure data folder exists if requested
    if "collect_data" in sys.argv:
        if not os.path.exists("lib/data"):
            os.makedirs("lib/data")

    # start the key listener in a background thread
    listener = keyboard.Listener(on_release=on_release)
    listener.daemon = True
    listener.start()

    # start the demo
    main()
