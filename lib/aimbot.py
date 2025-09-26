# lunar_safe.py  -- Non-interactive detection/demo only (no mouse control / no clicking)
import json
import math
import mss
import numpy as np
import os
import sys
import time
import torch
import uuid
import cv2

from termcolor import colored

class AimbotDemo:
    """
    Safe demo/refactor of the original script:
    - Uses YOLO model to detect players
    - Draws boxes, head markers, FPS
    - DOES NOT move the mouse or click (no SendInput / no mouse_event)
    - Can save frames for dataset collection when collect_data=True
    """

    screen = mss.mss()

    def __init__(self, box_constant=416, collect_data=False, mouse_delay=0.0001, debug=False):
        self.box_constant = int(box_constant)
        self.collect_data = bool(collect_data)
        self.mouse_delay = float(mouse_delay)
        self.debug = bool(debug)

        # load config file
        cfg_path = "lib/config/config.json"
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                self.sens_config = json.load(f)
        else:
            self.sens_config = {"targeting_scale": 1.0}

        print("[INFO] Loading the neural network model (ultralytics YOLOv5)")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='lib/best.pt', force_reload=True)
        if torch.cuda.is_available():
            print(colored("CUDA ACCELERATION [ENABLED]", "green"))
        else:
            print(colored("[!] CUDA ACCELERATION IS UNAVAILABLE", "red"))
            print(colored("[!] Check your PyTorch installation; performance will be slower", "red"))

        self.model.conf = 0.45
        self.model.iou = 0.45

        self.enabled = True  # demo detection toggle
        print("\n[INFO] PRESS 'F1' TO TOGGLE DETECTION\n[INFO] PRESS 'F2' OR 'ESC' TO QUIT\n[INFO] PRESS 'S' TO SAVE A FRAME (if collect_data enabled)")

    @staticmethod
    def interpolate_coordinates_from_center(absolute_coordinates, pixel_increment=1, scale=1.0, center=(960, 540)):
        """
        Generator that yields integer relative (x,y) steps from center to the absolute_coordinates.
        This is kept purely computational — NOT used to move the mouse.
        """
        diff_x = (absolute_coordinates[0] - center[0]) * scale / pixel_increment
        diff_y = (absolute_coordinates[1] - center[1]) * scale / pixel_increment
        length = int(math.dist((0, 0), (diff_x, diff_y)))
        if length == 0:
            return
        unit_x = (diff_x / length) * pixel_increment
        unit_y = (diff_y / length) * pixel_increment
        x = y = sum_x = sum_y = 0
        for k in range(0, length):
            sum_x += x
            sum_y += y
            x, y = round(unit_x * k - sum_x), round(unit_y * k - sum_y)
            yield x, y

    def _is_own_player(self, x1, y1, x2, y2):
        # same heuristic as original to try and ignore self model (best-effort)
        width = self.box_constant
        return x1 < 15 or (x1 < width / 5 and y2 > width / 1.2)

    def start(self):
        print("[INFO] Beginning screen capture (demo mode, no mouse control)")
        half_w = int(ctypes_safe_get_system_metric(0) / 2)
        half_h = int(ctypes_safe_get_system_metric(1) / 2)

        detection_box = {
            'left': int(half_w - self.box_constant // 2),
            'top': int(half_h - self.box_constant // 2),
            'width': int(self.box_constant),
            'height': int(self.box_constant)
        }

        collect_pause = 0.0
        window_name = "Lunar Vision (SAFE DEMO)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                loop_start = time.perf_counter()
                frame_bgra = np.array(AimbotDemo.screen.grab(detection_box))
                frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                orig_frame = frame.copy()

                results = self.model(frame)

                player_in_frame = False
                closest_detection = None
                least_crosshair_dist = None

                if len(results.xyxy[0]) != 0:
                    for *box, conf, cls in results.xyxy[0]:
                        x1y1 = [int(x.item()) for x in box[:2]]
                        x2y2 = [int(x.item()) for x in box[2:]]
                        x1, y1, x2, y2 = *x1y1, *x2y2
                        conf_val = float(conf.item())
                        height = y2 - y1
                        relative_head_X = int((x1 + x2) / 2)
                        relative_head_Y = int((y1 + y2) / 2 - height / 2.7)

                        own_player = self._is_own_player(x1, y1, x2, y2)

                        # distance to box center
                        center_x = self.box_constant // 2
                        center_y = self.box_constant // 2
                        crosshair_dist = math.dist((relative_head_X, relative_head_Y), (center_x, center_y))

                        if least_crosshair_dist is None:
                            least_crosshair_dist = crosshair_dist

                        if not own_player and (crosshair_dist <= least_crosshair_dist):
                            least_crosshair_dist = crosshair_dist
                            closest_detection = {
                                "x1y1": x1y1,
                                "x2y2": x2y2,
                                "relative_head_X": relative_head_X,
                                "relative_head_Y": relative_head_Y,
                                "conf": conf_val
                            }

                        # draw boxes for non-own players
                        if not own_player:
                            cv2.rectangle(frame, tuple(x1y1), tuple(x2y2), (244, 113, 115), 2)
                            cv2.putText(frame, f"{int(conf_val * 100)}%", tuple(x1y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (244, 113, 116), 2)
                        else:
                            player_in_frame = True

                # draw the chosen detection and debug information
                if closest_detection:
                    rx, ry = closest_detection["relative_head_X"], closest_detection["relative_head_Y"]
                    # marker on the detected head (relative coords inside the box)
                    cv2.circle(frame, (rx, ry), 5, (115, 244, 113), -1)
                    cv2.line(frame, (rx, ry), (self.box_constant // 2, self.box_constant // 2), (244, 242, 113), 2)

                    abs_x = rx + detection_box['left']
                    abs_y = ry + detection_box['top']
                    x1, y1 = closest_detection["x1y1"]

                    # print detection summary to console (useful for dataset or research)
                    print(f"[DETECT] head_abs=({abs_x},{abs_y}) head_rel=({rx},{ry}) conf={closest_detection['conf']:.2f}")

                    # overlay text
                    cv2.putText(frame, "TARGETING", (x1 + 40, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (115, 113, 244), 2)

                # FPS overlay
                fps = int(1.0 / (time.perf_counter() - loop_start)) if time.perf_counter() - loop_start > 0 else 0
                cv2.putText(frame, f"FPS: {fps}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
                cv2.imshow(window_name, frame)

                # keyboard controls via cv2.waitKey
                key = cv2.waitKey(1) & 0xFF
                # F1 toggle (common F1 code from cv2 isn't universal; accept 't' or lower-case 'f' as alternate toggle)
                if key == 0x70:  # F1 virtual-key code (works sometimes)
                    self.enabled = not self.enabled
                    print(f"[INFO] Detection {'ENABLED' if self.enabled else 'DISABLED'}")
                if key in (ord('t'), ord('T')):
                    self.enabled = not self.enabled
                    print(f"[INFO] Detection {'ENABLED' if self.enabled else 'DISABLED'}")

                if key in (ord('s'), ord('S')) and self.collect_data:
                    # save original frame for dataset
                    save_path = f"lib/data/{str(uuid.uuid4())}.jpg"
                    cv2.imwrite(save_path, orig_frame)
                    print(f"[INFO] Saved frame to: {save_path}")

                # F2 or ESC to quit
                if key == 0x71 or key == 27 or key == ord('q'):  # F2, ESC, or 'q'
                    print("[INFO] Quit requested.")
                    break

        finally:
            cv2.destroyAllWindows()
            AimbotDemo._safe_exit()

    @staticmethod
    def _safe_exit():
        print("[INFO] Exiting demo.")
        try:
            AimbotDemo.screen.close()
        except Exception:
            pass
        sys.exit(0)

def ctypes_safe_get_system_metric(index):
    """
    Lightweight wrapper without importing win32api.
    We'll try to call user32 GetSystemMetrics via ctypes.
    """
    try:
        import ctypes
        return ctypes.windll.user32.GetSystemMetrics(index)
    except Exception:
        # fallback to 1920x1080 assumptions if call fails
        return 1920 if index == 0 else 1080

if __name__ == "__main__":
    # Example: start demo with data-collection off
    demo = AimbotDemo(box_constant=416, collect_data=False, debug=False)
    demo.start()
