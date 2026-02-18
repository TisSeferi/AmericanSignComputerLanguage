import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp_solutions
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path

HOME      = Path(__file__).resolve().parent.parent
TEMPLATES = HOME / 'templates'

CANVAS_W = 640
CANVAS_H = 480

# MediaPipe hand skeleton
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Palette (matches main.py) ──────────────────────────────────────────────
BG         = '#12131a'
CARD_BG    = '#1e2130'
CONSOLE_BG = '#0a0b10'
CONSOLE_FG = '#69ff91'
MUTED_FG   = '#7986cb'
BTN_BG     = '#283593'
BTN_FG     = '#e8eaf6'
BTN_HOVER  = '#3949ab'
REC_RED    = '#ef5350'
BONE_COLOR = '#3949ab'
JOINT_COLOR= '#69ff91'


class TemplateCrafter:
    def __init__(self, root):
        self.root = root
        self.root.title('ASL Template Crafter')
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # State
        self.recording       = False
        self.gesture_data    = []       # list of (63,) ndarrays while recording
        self.current_gesture = None     # (n_frames, 63) ndarray after stop
        self.playback_mode   = False
        self.play_job        = None
        self.current_frame   = 0
        self.start_frame     = None
        self.end_frame       = None

        self._build_ui()

        # Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CANVAS_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)

        # MediaPipe
        self.hands      = mp_solutions.solutions.hands.Hands()
        self.mp_drawing = mp_solutions.solutions.drawing_utils

        self.update_frame()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        wrap = tk.Frame(self.root, bg='#000000', padx=2, pady=2)
        wrap.pack(padx=10, pady=(10, 0))
        self.canvas = tk.Canvas(wrap, width=CANVAS_W, height=CANVAS_H,
                                bg='#000000', highlightthickness=0)
        self.canvas.pack()

        # Status bar
        status_bar = tk.Frame(self.root, bg=CARD_BG, padx=10, pady=6)
        status_bar.pack(fill=tk.X, padx=10, pady=(4, 0))

        self._rec_dot = tk.Label(status_bar, text='●', font=('Helvetica', 13),
                                  fg=BG, bg=CARD_BG)
        self._rec_dot.pack(side=tk.LEFT)

        self._frame_count = tk.Label(status_bar, text='',
                                      font=('Courier', 10),
                                      fg=MUTED_FG, bg=CARD_BG)
        self._frame_count.pack(side=tk.LEFT, padx=(4, 0))

        self._status = tk.Label(status_bar, text='Ready',
                                 font=('Helvetica', 10),
                                 fg=CONSOLE_FG, bg=CARD_BG, anchor='e')
        self._status.pack(side=tk.RIGHT)

        # Buttons
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(padx=10, pady=8)

        def _btn(parent, label, cmd):
            b = tk.Button(
                parent, text=label, command=cmd,
                font=('Helvetica', 10, 'bold'),
                fg=BTN_FG, bg=BTN_BG,
                activebackground=BTN_HOVER, activeforeground=BTN_FG,
                relief=tk.FLAT, padx=14, pady=6, cursor='hand2', bd=0,
            )
            b.pack(side=tk.LEFT, padx=4)
            return b

        _btn(btn_row, 'Record',   self.start_recording)
        _btn(btn_row, 'Stop',     self.stop_recording)
        self._play_btn = _btn(btn_row, 'Playback', self.toggle_playback)
        _btn(btn_row, 'Save',     self.save_gesture)
        _btn(btn_row, 'Load',     self.load_gesture)

        # Scrubber row
        scrub_row = tk.Frame(self.root, bg=CARD_BG, padx=10, pady=8)
        scrub_row.pack(fill=tk.X, padx=10)

        self._slider = tk.Scale(
            scrub_row, from_=0, to=0, orient=tk.HORIZONTAL,
            command=self._on_slider,
            bg=CARD_BG, fg=MUTED_FG, troughcolor=BG,
            highlightthickness=0, showvalue=False,
            sliderlength=16, length=490,
        )
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._pb_info = tk.Label(scrub_row, text='—',
                                  font=('Courier', 10),
                                  fg=MUTED_FG, bg=CARD_BG, width=14)
        self._pb_info.pack(side=tk.LEFT, padx=(8, 0))

        # Marker row
        marker_row = tk.Frame(self.root, bg=BG)
        marker_row.pack(padx=10, pady=(4, 10))

        _btn(marker_row, '[E]  Set Start', self.set_start)
        _btn(marker_row, '[R]  Set End',   self.set_end)

        self._marker_label = tk.Label(marker_row, text='',
                                       font=('Courier', 10),
                                       fg=MUTED_FG, bg=BG)
        self._marker_label.pack(side=tk.LEFT, padx=10)

        # Keybindings
        self.root.bind('<space>', lambda e: self.toggle_playback())
        self.root.bind('<Left>',  lambda e: self._step(-1))
        self.root.bind('<Right>', lambda e: self._step(1))
        self.root.bind('e',       lambda e: self.set_start())
        self.root.bind('r',       lambda e: self.set_end())

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, msg):
        self._status.config(text=msg)

    def _update_marker_label(self):
        parts = []
        if self.start_frame is not None:
            parts.append(f'Start: {self.start_frame}')
        if self.end_frame is not None:
            parts.append(f'End: {self.end_frame}')
        self._marker_label.config(text='   '.join(parts) if parts else '')

    def _update_pb_info(self):
        if self.current_gesture is not None:
            n = len(self.current_gesture)
            self._pb_info.config(text=f'{self.current_frame + 1} / {n}')

    # ── Camera loop ───────────────────────────────────────────────────────────
    def update_frame(self):
        if self.playback_mode:
            self.root.after(16, self.update_frame)
            return

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame_rgb, hand_lms,
                        mp_solutions.solutions.hands.HAND_CONNECTIONS,
                    )
                    if self.recording:
                        lms = hand_lms.landmark
                        pt  = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                                       dtype=np.float64).reshape(-1)
                        self.gesture_data.append(pt)
                        self._frame_count.config(text=f'{len(self.gesture_data)} frames')

            img   = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.root.after(10, self.update_frame)

    # ── Recording ─────────────────────────────────────────────────────────────
    def start_recording(self):
        self.playback_mode   = False
        self.recording       = True
        self.gesture_data    = []
        self.start_frame     = None
        self.end_frame       = None
        self._update_marker_label()
        self._rec_dot.config(fg=REC_RED)
        self._frame_count.config(text='0 frames')
        self._set_status('Recording…')

    def stop_recording(self):
        if not self.recording:
            return
        self.recording       = False
        self.current_gesture = np.array(self.gesture_data, dtype=np.float64)
        n = len(self.current_gesture)
        self._rec_dot.config(fg=BG)
        self._frame_count.config(text='')
        self._set_status(f'Captured {n} frames — ready to review')
        if n > 0:
            self._slider.config(to=n - 1)
            self.current_frame = 0
            self._slider.set(0)
            self._update_pb_info()

    # ── Playback ──────────────────────────────────────────────────────────────
    def toggle_playback(self):
        if self.current_gesture is None or len(self.current_gesture) == 0:
            self._set_status('Nothing recorded yet.')
            return
        if self.playback_mode:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self.playback_mode = True
        self._play_btn.config(text='Live')
        self._set_status('Playback  ·  Space  ←/→  E/R to mark start/end')
        self._draw_skeleton(self.current_frame)
        self._advance()

    def _stop_playback(self):
        self.playback_mode = False
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self._play_btn.config(text='Playback')
        self._set_status('Ready')

    def _advance(self):
        if not self.playback_mode:
            return
        n    = len(self.current_gesture)
        next = (self.current_frame + 1) % n
        self._slider.set(next)          # triggers _on_slider → draws frame
        self.play_job = self.root.after(50, self._advance)  # ~20 fps

    def _on_slider(self, val):
        self.current_frame = int(float(val))
        self._draw_skeleton(self.current_frame)
        self._update_pb_info()

    def _step(self, delta):
        if self.current_gesture is None:
            return
        n = len(self.current_gesture)
        self.current_frame = max(0, min(n - 1, self.current_frame + delta))
        self._slider.set(self.current_frame)

    # ── Skeleton renderer ─────────────────────────────────────────────────────
    def _draw_skeleton(self, idx):
        self.canvas.delete('all')
        if self.current_gesture is None or idx >= len(self.current_gesture):
            return

        pts = self.current_gesture[idx].reshape(21, 3)
        xs  = (pts[:, 0] * CANVAS_W).astype(int)
        ys  = (pts[:, 1] * CANVAS_H).astype(int)

        for a, b in CONNECTIONS:
            self.canvas.create_line(xs[a], ys[a], xs[b], ys[b],
                                    fill=BONE_COLOR, width=2)
        r = 4
        for i in range(21):
            self.canvas.create_oval(xs[i]-r, ys[i]-r, xs[i]+r, ys[i]+r,
                                    fill=JOINT_COLOR, outline='')

        n = len(self.current_gesture)
        self.canvas.create_text(8, 8, anchor='nw',
                                 text=f'Frame {idx + 1} / {n}',
                                 fill=MUTED_FG, font=('Courier', 10))
        if self.start_frame == idx:
            self.canvas.create_text(8, 26, anchor='nw', text='▶ START',
                                     fill=CONSOLE_FG, font=('Courier', 9, 'bold'))
        if self.end_frame == idx:
            self.canvas.create_text(8, 42, anchor='nw', text='■ END',
                                     fill=REC_RED, font=('Courier', 9, 'bold'))

    # ── Markers ───────────────────────────────────────────────────────────────
    def set_start(self):
        self.start_frame = self.current_frame
        self._update_marker_label()
        self._set_status(f'Start frame: {self.start_frame}')
        if self.playback_mode:
            self._draw_skeleton(self.current_frame)

    def set_end(self):
        self.end_frame = self.current_frame
        self._update_marker_label()
        self._set_status(f'End frame: {self.end_frame}')
        if self.playback_mode:
            self._draw_skeleton(self.current_frame)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save_gesture(self):
        if self.current_gesture is None or len(self.current_gesture) == 0:
            self._set_status('Nothing to save.')
            return

        if self.start_frame is not None and self.end_frame is not None:
            lo   = min(self.start_frame, self.end_frame)
            hi   = max(self.start_frame, self.end_frame)
            data = self.current_gesture[lo:hi + 1]
        else:
            data = self.current_gesture

        TEMPLATES.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(TEMPLATES),
            defaultextension='.npy',
            filetypes=[('NumPy array', '*.npy')],
        )
        if path:
            np.save(path, data)
            self._set_status(f'Saved {len(data)} frames → {Path(path).name}')

    def load_gesture(self):
        path = filedialog.askopenfilename(
            initialdir=str(TEMPLATES),
            filetypes=[('NumPy array', '*.npy')],
        )
        if path:
            self.current_gesture = np.load(path).astype(np.float64)
            n = len(self.current_gesture)
            self.start_frame     = None
            self.end_frame       = None
            self.current_frame   = 0
            self._slider.config(to=n - 1)
            self._slider.set(0)
            self._update_marker_label()
            self._update_pb_info()
            self._set_status(f'Loaded {n} frames ← {Path(path).name}')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def close(self):
        self._stop_playback()
        self.cap.release()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app  = TemplateCrafter(root)
    root.protocol('WM_DELETE_WINDOW', app.close)
    root.mainloop()
