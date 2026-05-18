"""
fitness_trainer.py — Fitness trainer app using DIPPID + pyglet.
Trains a classifier on startup, then guides the user through activities
and checks in real time whether they are executing them correctly.
"""

import os
import random
import pyglet
from pyglet import shapes
from DIPPID import SensorUDP
import activity_recognizer as ar

PORT = 5700
ACTIVITY_DURATION = 20   
COUNTDOWN = 3             

SHARED_DATA = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assignment-03-training-data-join-this-team-to-upload-your-data",
)

recognizer = ar.ActivityRecognizer(data_dir=SHARED_DATA)

print("Starting Fitness Trainer...")
try:
    recognizer.train()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    print("No training data found. Classifier will not work until data is gathered.")

sensor = SensorUDP(PORT)

W, H = 800, 600
window = pyglet.window.Window(W, H, caption="Fitness Trainer")
batch = pyglet.graphics.Batch()

# State
state = {
    "phase": "idle",          # idle | countdown | active | done
    "target": None,           # activity to perform
    "countdown": COUNTDOWN,
    "elapsed": 0.0,
    "prediction": "—",
    "correct_streak": 0,
    "queue": list(ar.ACTIVITIES),
    "success": False,
}
random.shuffle(state["queue"])

COLORS = {
    "running":      (52, 152, 219),
    "rowing":       (46, 204, 113),
    "lifting":      (231, 76, 60),
    "jumpingjacks": (155, 89, 182),
}

lbl_title = pyglet.text.Label(
    "Fitness Trainer",
    font_name="Arial Bold", font_size=28,
    x=W // 2, y=H - 50, anchor_x="center", anchor_y="center",
    color=(255, 255, 255, 255), batch=batch,
)
lbl_activity = pyglet.text.Label(
    "Press button_1 to start",
    font_name="Arial", font_size=22,
    x=W // 2, y=H // 2 + 60, anchor_x="center", anchor_y="center",
    color=(255, 220, 0, 255), batch=batch,
)
lbl_instruction = pyglet.text.Label(
    "",
    font_name="Arial", font_size=16,
    x=W // 2, y=H // 2, anchor_x="center", anchor_y="center",
    color=(200, 200, 200, 255), batch=batch,
)
lbl_prediction = pyglet.text.Label(
    "Detected: —",
    font_name="Arial", font_size=18,
    x=W // 2, y=H // 2 - 70, anchor_x="center", anchor_y="center",
    color=(100, 255, 100, 255), batch=batch,
)
lbl_status = pyglet.text.Label(
    "",
    font_name="Arial Bold", font_size=20,
    x=W // 2, y=H // 2 - 140, anchor_x="center", anchor_y="center",
    color=(255, 255, 255, 255), batch=batch,
)
lbl_progress = pyglet.text.Label(
    "",
    font_name="Arial", font_size=13,
    x=W // 2, y=40, anchor_x="center", anchor_y="center",
    color=(180, 180, 180, 255), batch=batch,
)

bg_rect = shapes.Rectangle(0, 0, W, H, color=(30, 30, 40), batch=batch)
bg_rect.group = pyglet.graphics.Group(order=-1)

indicator = shapes.Circle(W // 2, H // 2 - 200, 18, color=(80, 80, 80), batch=batch)


def start_next_activity():
    if not state["queue"]:
        state["phase"] = "done"
        return
    state["target"] = state["queue"].pop(0)
    state["phase"] = "countdown"
    state["countdown"] = COUNTDOWN
    state["elapsed"] = 0.0
    state["correct_streak"] = 0


def on_button(data):
    # DIPPID Android sends 0 for both press and release — trigger on any event
    if state["phase"] == "idle":
        start_next_activity()


sensor.register_callback("button_1", on_button)


_latest_gyro = {"x": 0.0, "y": 0.0, "z": 0.0}


def on_gyro_data(data):
    if isinstance(data, dict) and "x" in data:
        _latest_gyro.update(data)


def on_acc_data(data):
    # Called only when a NEW accelerometer packet arrives (~11 Hz on Android DIPPID)
    if not isinstance(data, dict) or "x" not in data:
        return
    if state["phase"] != "active":
        return
    pred = recognizer.update(data, dict(_latest_gyro))
    if pred:
        state["prediction"] = pred


sensor.register_callback("accelerometer", on_acc_data)
sensor.register_callback("gyroscope", on_gyro_data)


@window.event
def on_draw():
    window.clear()
    phase = state["phase"]
    target = state["target"]
    if phase == "active":
        correct = state["prediction"] == target
        bg_rect.color = (20, 60, 20) if correct else (60, 20, 20)
        indicator.color = (50, 255, 50) if correct else (255, 50, 50)
    else:
        bg_rect.color = (30, 30, 40)
        indicator.color = (80, 80, 80)

    if phase == "idle":
        lbl_activity.text = "Fitness Trainer"
        lbl_instruction.text = "Press button_1 on DIPPID to begin"
        lbl_prediction.text = ""
        lbl_status.text = ""

    elif phase == "countdown":
        lbl_activity.text = f"Get ready: {target.upper()}"
        lbl_instruction.text = f"Starting in {int(state['countdown']) + 1}..."
        lbl_prediction.text = ""
        lbl_status.text = ""

    elif phase == "active":
        remaining = max(0, ACTIVITY_DURATION - state["elapsed"])
        correct = state["prediction"] == target
        lbl_activity.text = f"DO: {target.upper()}"
        lbl_instruction.text = f"Time remaining: {remaining:.1f}s"
        lbl_prediction.text = f"Detected: {state['prediction']}"
        lbl_status.text = "✓ Correct!" if correct else "✗ Wrong activity"
        lbl_status.color = (50, 255, 50, 255) if correct else (255, 80, 80, 255)

    elif phase == "done":
        lbl_activity.text = "Workout complete! 🎉"
        lbl_instruction.text = "Great job!"
        lbl_prediction.text = ""
        lbl_status.text = ""

    done_count = len(ar.ACTIVITIES) - len(state["queue"]) - (1 if phase == "active" else 0)
    lbl_progress.text = f"Activities: {done_count}/{len(ar.ACTIVITIES)}"

    batch.draw()


def update(dt):
    phase = state["phase"]
    if phase == "countdown":
        state["countdown"] -= dt
        if state["countdown"] <= 0:
            state["phase"] = "active"
            state["elapsed"] = 0.0

    elif phase == "active":
        state["elapsed"] += dt
        if state["prediction"] == state["target"]:
            state["correct_streak"] += dt
        if state["elapsed"] >= ACTIVITY_DURATION:
            start_next_activity()


pyglet.clock.schedule_interval(update, 1 / 60)

print(f"Listening on UDP port {PORT}. Press button_1 on DIPPID to start.")
pyglet.app.run()
