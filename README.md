[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/CjRQqtHi)

# Assignment 3 — Activity Recognition

Fitness trainer application that uses a DIPPID sensor to recognize physical activities in real time via a machine learning classifier.

## Activities

`running` · `rowing` · `lifting` · `jumpingjacks`

## Setup

```bash
pip install pandas numpy scikit-learn pyglet DIPPID
```

## Usage

### 1. Gather training data

```bash
python gather_data.py <name> <activity> <number>
# e.g.
python gather_data.py jonas running 1
```

- Connect the DIPPID device and press any button to start a 10-second recording.
- Data is saved to `data/<name>-<activity>-<number>.csv`.
- Record at least 5 files per activity.

### 2. Run the fitness trainer

```bash
python fitness_trainer.py
```

On startup the classifier is trained on all CSV files in `data/` and the shared course dataset. Training accuracy is printed to the console. Press `button_1` on the DIPPID device to begin the workout session.

## How it works

| Step | Details |
|---|---|
| Feature extraction | Sliding window (1 s, 50 % overlap) over acc + gyro signals; mean, std, min, max, range + top-5 FFT magnitudes per axis → 60 features |
| Classifier | Random Forest (100 trees), trained on an 80/20 train/test split |
| Live prediction | Each new sensor sample is buffered; a prediction is emitted every 0.5 s (STEP\_SIZE samples) |
| Test accuracy | ~97 % when trained on the full shared dataset (325 recordings, 6 000+ windows) |

## File overview

| File | Purpose |
|---|---|
| `gather_data.py` | Records and saves sensor data as CSV |
| `activity_recognizer.py` | Feature extraction, model training, live prediction |
| `fitness_trainer.py` | Pyglet GUI — guides user through activities and shows real-time feedback |
| `DIPPID.py` | DIPPID UDP sensor library |
| `data/` | Training recordings (local) |
