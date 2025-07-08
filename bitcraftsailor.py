import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageGrab
import cv2
import numpy as np
from collections import deque
import pyautogui
import threading
import random
import time
import sys

grid_offset_x = 0
grid_offset_y = 0
auto_navigate = False  # Toggle for auto-navigation
tile_cache = set()
last_known_player_pos = None
running = True
# Global cache to store tile states
tile_state_cache = {}



def capture_screen(region=None):
    screenshot = ImageGrab.grab(bbox=region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def classify_tiles(image, tile_w, tile_h, offset_x, offset_y):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    grid = {}

    for y in range(offset_y, h - tile_h, tile_h):
        for x in range(offset_x, w - tile_w, tile_w):
            cx = x + tile_w // 2
            cy = y + tile_h // 2
            pixel = rgb[cy, cx]

            if np.all(np.abs(pixel - [39, 39, 38]) <= 10):
                label = "FOG"
            elif np.all(np.abs(pixel - [26, 51, 76]) <= 20):
                label = "WATER"
            else:
                label = "LAND"

            grid[(x, y)] = label
            # Update the cache with the current label and timestamp
            tile_state_cache[(x, y)] = {'label': label, 'last_seen': time.time()}

    return grid

def draw_tiles_on_image(image, tiles):
    img_copy = image.copy()
    for (x, y, w, h) in tiles:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy

def get_reachable_fog_tiles(player_pos, grid, tile_w, tile_h):
    start = None
    for (x, y), label in grid.items():
        cx = x + tile_w // 2
        cy = y + tile_h // 2
        if abs(player_pos[0] - cx) <= tile_w//2 and abs(player_pos[1] - cy) <= tile_h//2:
            start = (x, y)
            break

    if not start:
        print("❌ Player not on grid tile")
        return []

    print(f"🌊 BFS Start Tile: {start} = {grid[start]}")
    visited = set()
    queue = deque([start])
    reachable_fog = []

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))

        label = grid.get((x, y))

        if label == "FOG" and (x, y) not in tile_cache:
            reachable_fog.append((x, y, tile_w, tile_h))

        if label not in ["WATER", "FOG"]:
            continue

        for dx in [-tile_w, 0, tile_w]:
            for dy in [-tile_h, 0, tile_h]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor not in visited and grid.get(neighbor) in ["FOG", "WATER"]:
                    queue.append(neighbor)

    # Include tiles from the cache that were previously reachable
    for (x, y), info in tile_state_cache.items():
        if info['label'] == "FOG" and (x, y) not in visited:
            reachable_fog.append((x, y, tile_w, tile_h))

    return reachable_fog


def detect_player_position(screen_img, icon_img_path="player_icon.png"):
    player_icon_orig = cv2.imread(icon_img_path)
    if player_icon_orig is None:
        print("❌ Could not load player icon image")
        return None

    best_match = None
    best_val = 0
    best_loc = (0, 0)

    for angle in range(0, 360, 15):
        M = cv2.getRotationMatrix2D((player_icon_orig.shape[1]//2, player_icon_orig.shape[0]//2), angle, 1)
        rotated_icon = cv2.warpAffine(player_icon_orig, M, (player_icon_orig.shape[1], player_icon_orig.shape[0]))
        result = cv2.matchTemplate(screen_img, rotated_icon, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_match = rotated_icon
            best_loc = max_loc

    if best_val < 0.7:
        print("⚠️ Player icon not confidently found")
        return None

    icon_w, icon_h = best_match.shape[1], best_match.shape[0]
    center_x = best_loc[0] + icon_w // 2
    center_y = best_loc[1] + icon_h // 2

    print(f"🧭 Player detected at: ({center_x}, {center_y}) | confidence={best_val:.2f}")
    return (center_x, center_y)


def find_nearest_tile(player_pos, tiles):
    px, py = player_pos
    def tile_center(tile):
        x, y, w, h = tile
        return (x + w//2, y + h//2)
    return min(tiles, key=lambda t: (tile_center(t)[0] - px) ** 2 + (tile_center(t)[1] - py) ** 2)

def find_path(start, goal, grid, tile_w, tile_h):
    queue = deque()
    queue.append((start, [start]))
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path

        x, y = current
        neighbors = [
            (x - tile_w, y),
            (x + tile_w, y),
            (x, y - tile_h),
            (x, y + tile_h),
            (x - tile_w, y - tile_h),
            (x + tile_w, y - tile_h),
            (x - tile_w, y + tile_h),
            (x + tile_w, y + tile_h)
        ]

        for neighbor in neighbors:
            if neighbor in grid and grid[neighbor] in ["FOG", "WATER"] and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []

def draw_path_on_image(image, path, tile_w, tile_h):
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        center1 = (x1 + tile_w // 2, y1 + tile_h // 2)
        center2 = (x2 + tile_w // 2, y2 + tile_h // 2)
        cv2.line(image, center1, center2, (255, 255, 0), 2)
    return image

def move_to_tile(tile):
    x, y, w, h = tile
    center_x = x + w // 2
    center_y = y + h // 2
    pyautogui.moveTo(center_x, center_y)
    pyautogui.click()

def update_image():
    bgr_image = capture_screen()
    player_pos = detect_player_position(bgr_image)

    if not player_pos:
        status_label.config(text="⚠️ Player not found")
        return

    grid = classify_tiles(bgr_image, 50, 45, grid_offset_x, grid_offset_y)
    all_fog_tiles = [(x, y, 50, 45) for (x, y), label in grid.items() if label == "FOG"]
    print(f"🟫 Total fog tiles classified: {len(all_fog_tiles)}")

    tiles = get_reachable_fog_tiles(player_pos, grid, 50, 45)
    print(f"✅ Reachable fog tiles: {len(tiles)}")

    marked_image = draw_tiles_on_image(bgr_image, tiles)
    cv2.circle(marked_image, player_pos, 10, (255, 0, 0), 3)  # Player

    if tiles:
        nearest_tile = find_nearest_tile(player_pos, tiles)
        x, y, w, h = nearest_tile
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red box
        print(f"🎯 Nearest reachable tile: ({x}, {y})")

        # Find path
        start_tile = None
        for (tx, ty), label in grid.items():
            cx = tx + w // 2
            cy = ty + h // 2
            if abs(player_pos[0] - cx) <= w // 2 and abs(player_pos[1] - cy) <= h // 2:
                start_tile = (tx, ty)
                break

        if start_tile:
            path = find_path(start_tile, (x, y), grid, w, h)
            print(f"📏 Path length: {len(path)} steps")
            marked_image = draw_path_on_image(marked_image, path, w, h)

            if auto_navigate:
                time.sleep(2)  # Delay before auto-move
                move_to_tile(nearest_tile)

    rgb_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(img.resize((800, 600)))

    # Maintain a reference to the image to prevent garbage collection
    image_label.image = tk_image
    image_label.configure(image=tk_image)
    status_label.config(text=f"Reachable fog: {len(tiles)}")

def draw_tiles_on_image(image, tiles):
    img_copy = image.copy()
    for (x, y, w, h) in tiles:
        # Use a different color for cached tiles
        color = (0, 255, 0) if (x, y) in tile_state_cache else (0, 100, 0)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
    return img_copy

def calibrate_offset(event):
    global grid_offset_x, grid_offset_y
    grid_offset_x = int(event.x * image_label.image.width() / image_label.winfo_width())
    grid_offset_y = int(event.y * image_label.image.height() / image_label.winfo_height())
    print(f"✅ Grid offset set to: ({grid_offset_x}, {grid_offset_y})")
    update_image()

def toggle_auto_nav():
    global auto_navigate
    auto_navigate = not auto_navigate
    toggle_btn.config(text=f"Auto-Navigate: {'ON' if auto_navigate else 'OFF'}")
    print(f"🧭 Auto-Navigation is now {'enabled' if auto_navigate else 'disabled'}")

def navigate_next_tile():
    if not auto_navigate or not running:
        return

    bgr_image = capture_screen()
    player_pos = detect_player_position(bgr_image)
    if not player_pos:
        print("⚠️ Player not found for navigation")
        schedule_next_navigation()
        return

    grid = classify_tiles(bgr_image, 50, 45, grid_offset_x, grid_offset_y)
    reachable = get_reachable_fog_tiles(player_pos, grid, 50, 45)
    if not reachable:
        print("❌ No reachable fog tiles")
        schedule_next_navigation()
        return

    nearest = find_nearest_tile(player_pos, reachable)
    x, y, w, h = nearest

    start_tile = None
    for (tx, ty), label in grid.items():
        cx = tx + w // 2
        cy = ty + h // 2
        if abs(player_pos[0] - cx) <= w // 2 and abs(player_pos[1] - cy) <= h // 2:
            start_tile = (tx, ty)
            break

    if not start_tile:
        print("⚠️ Couldn't align player to tile grid")
        schedule_next_navigation()
        return

    path = find_path(start_tile, (x, y), grid, w, h)
    if len(path) < 2:
        print("🔹 Already at or near destination")
        tile_cache.add((x, y))  # Add to cache if unreachable
        schedule_next_navigation()
        return

    next_tile = path[1]
    center_x = next_tile[0] + w // 2
    center_y = next_tile[1] + h // 2
    print(f"🖱️ Clicking next tile: {center_x}, {center_y}")

    pyautogui.moveTo(center_x, center_y, duration=0.3)
    pyautogui.click()

    # Add the clicked tile to the cache
    tile_cache.add((x, y))

    schedule_next_navigation()


def schedule_next_navigation():
    delay = random.randint(10, 18)
    print(f"⏳ Scheduling next move in {delay} seconds")
    countdown(delay)
    threading.Timer(delay, navigate_next_tile).start()

def countdown(seconds):
    def tick():
        nonlocal seconds
        if seconds > 0 and auto_navigate:
            print(f"🕒 Next click in: {seconds}s")
            seconds -= 1
            threading.Timer(1, tick).start()
    tick()

def toggle_auto_nav():
    global auto_navigate
    auto_navigate = not auto_navigate
    toggle_btn.config(text=f"Auto-Navigate: {'ON' if auto_navigate else 'OFF'}")
    print(f"🧭 Auto-Navigation is now {'enabled' if auto_navigate else 'disabled'}")

    if auto_navigate:
        schedule_next_navigation()

def schedule_gui_refresh():
    update_image()
    root.after(10000, schedule_gui_refresh)  # every 10 seconds

def on_closing():
    global running
    running = False
    root.destroy()
    sys.exit()


# GUI Setup
root = tk.Tk()
root.title("Fog of War Detector")

frame = ttk.Frame(root, padding="10")
frame.grid()

image_label = ttk.Label(frame)
image_label.grid(row=0, column=0, columnspan=2)

status_label = ttk.Label(frame, text="Ready", padding="5")
status_label.grid(row=1, column=0, columnspan=2)

refresh_btn = ttk.Button(frame, text="Refresh Screenshot", command=update_image)
refresh_btn.grid(row=2, column=0, pady=10)

toggle_btn = ttk.Button(frame, text="Auto-Navigate: OFF", command=toggle_auto_nav)
toggle_btn.grid(row=2, column=1, pady=10)

image_label.bind("<Button-1>", calibrate_offset)
update_image()

schedule_gui_refresh()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
