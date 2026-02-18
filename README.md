# American Sign Computer Language (ASCL)

An interactive tool for recognizing and creating custom ASL hand gestures, supporting both dynamic (movement-based) and static (pose-based) recognition in real time.

---

## Overview

The recognizer combines two gesture recognition algorithms with MediaPipe hand tracking:

| Component | Role |
|---|---|
| **Jackknife** | DTW-based classification for dynamic gestures |
| **Machete** | Online segmentation for detecting gesture boundaries |
| **MediaPipe** | 21-point 3D hand landmark extraction |
| **NumPy / Numba** | Vectorized math and JIT-compiled DTW for performance |

---

## Project Structure

```
Scripts/
├── main.py               # GUI application (recognition mode)
├── TemplateCrafter.py    # GUI application (template recording mode)
├── mathematics.py        # Shared math utilities (DTW helpers, resampling, etc.)
├── jackknife/            # Jackknife recognizer package
│   ├── Jackknife.py
│   ├── JkBlades.py
│   ├── JkFeatures.py
│   ├── JkTemplate.py
│   └── JackknifeConnector.py
└── machete/              # Machete segmenter package
    ├── Machete.py
    ├── MacheteTemplate.py
    ├── MacheteElement.py
    ├── MacheteTrigger.py
    ├── MacheteSample.py
    ├── ContinuousResult.py
    └── CircularBuffer.py

templates/                # .npy gesture template files
```

---

## Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the recognizer**
```bash
python Scripts/main.py
```

**3. Record new gesture templates**
```bash
python Scripts/TemplateCrafter.py
```

---

## Usage

### Recognition (`main.py`)

- The app shows a random gesture from the template library and prompts you to perform it.
- Hold your hand in front of the camera.
- **Static gestures** — hold the pose; recognition fires after 3 consistent frames.
- **Dynamic gestures** — perform the movement naturally; Machete detects the start/end automatically.
- A green flash confirms a correct match; the next gesture is queued automatically.
- Press **Space** or click **Next →** to skip to a different gesture.

### Template Recording (`TemplateCrafter.py`)

- Choose static or dynamic mode before recording.
- Review the recording frame-by-frame before saving.
- Saved templates are written to `templates/` as `.npy` files and are picked up automatically on next launch.

---

## Architecture

```
Camera frame
    └─► MediaPipe → 21 landmarks → (63,) ndarray
            ├─► Machete (dynamic pipeline)
            │       └─► ContinuousResult → JKConnector → Jackknife.is_match()
            └─► Static worker
                    └─► joint angle cosine similarity against static templates
```

Processing runs across four `multiprocessing.Process` workers so the GUI thread stays responsive:

| Worker | Job |
|---|---|
| `process_frame_worker` | Feeds points into Machete's online DTW |
| `select_result_worker` | Picks the best candidate from Machete's result list |
| `match_worker` | Runs Jackknife DTW to confirm/reject a candidate |
| `static_worker` | Runs cosine-similarity matching for static poses |

---

## Performance

The DTW DP recurrence is JIT-compiled with **Numba** (`@njit(cache=True)`), giving roughly **100× speedup** over a pure-Python loop. All feature extraction and cost matrix computation uses NumPy vectorized operations.

---

## Dependencies

- `mediapipe` — hand landmark detection
- `opencv-python` / `opencv-contrib-python` — camera capture and image processing
- `numpy` — numerical computation
- `numba` — JIT compilation for the DTW inner loop
- `Pillow` — frame display in the Tkinter canvas

---

## Known Limitations

- Single-hand gestures only.
- Recognition accuracy depends on consistent lighting and camera quality.
- Template recording (`TemplateCrafter.py`) is a separate window from the main app.

---

## References

- [Jackknife (ISUE)](https://github.com/ISUE/Jackknife)
- [Machete (ISUE)](https://github.com/ISUE/Machete)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV](https://opencv.org/)
