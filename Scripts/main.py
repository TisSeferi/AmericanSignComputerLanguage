import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import cv2
import numpy as np
import multiprocessing as mp
from pathlib import Path
import collections as col
import re
from machete import Machete, ContinuousResult, ContinuousResultOptions
from jackknife import JKConnector as jkc, JkBlades, Jackknife
import mathematics
import pickle
from PIL import Image, ImageTk
import os
import mediapipe as mp_solutions
import random

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_POINTS     = 21
HOME           = Path(__file__).resolve().parent.parent
TEMPLATES      = HOME / 'templates'
RECOGNIZER_PKL = HOME / 'recognizer.pkl'


# ── Helpers ───────────────────────────────────────────────────────────────────
def gesture_family(template_name: str) -> str:
    """'ThankYou3_Left' → 'ThankYou',  'A1_Left' → 'A',  'W4' → 'W'"""
    name = re.sub(r'_(?:Left|Right)$', '', template_name)  # strip side suffix
    name = re.sub(r'\d+_*$', '', name).strip('_')           # strip instance digits
    return name


def display_name(family: str) -> str:
    """'ThankYou' → 'THANK YOU',  'OK' → 'OK'"""
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', family).upper()


def landmarks_to_frame(results):
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in lms]).reshape(-1)


def assemble_templates():
    templates = []
    for path in os.listdir(TEMPLATES):
        name = path.split('.')[0]
        templates.append((name, np.load(TEMPLATES / path)))
    return templates


# ── Workers (logic unchanged) ─────────────────────────────────────────────────
def process_frame_worker(machete, dynamic_queue, result_queue):
    while True:
        task = dynamic_queue.get()
        if task is None:
            break
        point, current_frame_num, ret = task
        machete.process_frame(point, current_frame_num, ret)
        result_queue.put((ret, point, current_frame_num))


def select_result_worker(result_queue, match_queue):
    while True:
        task = result_queue.get()
        if task is None:
            break
        ret, point, current_frame_num = task
        result = ContinuousResult.select_result(ret, False)
        match_queue.put((result, point, current_frame_num))


def match_worker(match_queue, recognizer_options, data_queue, output_queue):
    while True:
        task = match_queue.get()
        if task is None:
            break
        result, point, current_frame_num = task
        if result:
            data = data_queue.get()
            try:
                jk_buffer = jkc.get_jk_buffer_from_video(
                    list(data),
                    result.start_frame_no - current_frame_num,
                    result.end_frame_no - current_frame_num - 1
                )
                match, recognizer_d = recognizer_options.is_match(
                    trajectory=jk_buffer, gid=result.sample.gesture_id
                )
                if match:
                    output_queue.put(
                        f"Dynamic Gesture: {result.sample.gesture_id} | Score: {recognizer_d:.2f}"
                    )
            finally:
                data_queue.put(data)


def static_worker(static_queue, recognizer_options, output_queue):
    match_history = col.deque(maxlen=3)
    static_templates = [t for t in recognizer_options.templates if t.features.is_static]
    while True:
        task = static_queue.get()
        if task is None:
            break
        point, current_frame_num, ret = task
        point_centroid = mathematics.calculate_centroid(point)
        point_vecs_flat, _ = mathematics.convert_joint_positions_to_distance_vectors(
            point, point_centroid
        )
        best_match    = None
        best_distance = float('-inf')

        for template in static_templates:
            _, total_distance = mathematics.calculate_joint_angle_disparity(
                template.features.ff_joint_vecs_flat, point_vecs_flat
            )
            threshold = 0.9 * NUM_POINTS
            if total_distance > best_distance and total_distance > threshold:
                best_distance = total_distance
                best_match    = template

        if best_match:
            match_history.append(best_match.gesture_id)
            if len(match_history) == 3 and len(set(match_history)) == 1:
                output_queue.put(
                    f"Static Gesture: {best_match.gesture_id} | Score: {best_distance:.2f}"
                )
                match_history.clear()


# ── GUI ───────────────────────────────────────────────────────────────────────
class GestureApp:
    # Palette
    BG          = '#12131a'   # window background
    CARD_BG     = '#1e2130'   # log panel
    INSTR_BG    = '#1565c0'   # instruction card (blue)
    INSTR_DIM   = '#90caf9'   # instruction subtitle
    INSTR_FG    = '#ffffff'   # instruction gesture name
    SUCCESS_BG  = '#1b5e20'   # instruction card on correct match (green)
    SUCCESS_FG  = '#a5d6a7'   # gesture name during success flash
    SUCCESS_DIM = '#4caf50'   # subtitle during success flash
    CONSOLE_BG  = '#0a0b10'   # log text area
    CONSOLE_FG  = '#69ff91'   # terminal-green log text
    MUTED_FG    = '#7986cb'   # secondary labels
    BTN_BG      = '#283593'   # next-gesture button
    BTN_FG      = '#e8eaf6'
    BTN_HOVER   = '#3949ab'

    def __init__(self, root):
        self.root = root
        self.root.title("ASL Gesture Recognizer")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.current_gesture_family = None
        self._flashing   = False
        self._flash_job  = None

        self._build_ui()

        # Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self._log_raw("Error: cannot open webcam.")

        self.hands = mp_solutions.solutions.hands.Hands()

        # Queues
        self.dynamic_queue = mp.Queue()
        self.static_queue  = mp.Queue()
        self.result_queue  = mp.Queue()
        self.match_queue   = mp.Queue()
        self.output_queue  = mp.Queue()
        self.data_queue    = mp.Queue()
        self.data_queue.put(col.deque(maxlen=100))

        # Recognizer + templates
        raw = assemble_templates()
        self._families = sorted(set(gesture_family(n) for n, _ in raw))

        self.machete = Machete(
            device_type=None,
            cr_options=ContinuousResultOptions(),
            templates=raw,
        )
        self.blades = JkBlades()
        self.blades.set_ip_defaults()

        try:
            with open(RECOGNIZER_PKL, 'rb') as f:
                self.recognizer_options = pickle.load(f)
        except Exception:
            self.recognizer_options = Jackknife(templates=raw, blades=self.blades)
            with open(RECOGNIZER_PKL, 'wb') as f:
                pickle.dump(self.recognizer_options, f)

        # Workers
        self.frame_worker = mp.Process(
            target=process_frame_worker,
            args=(self.machete, self.dynamic_queue, self.result_queue),
        )
        self.result_worker = mp.Process(
            target=select_result_worker,
            args=(self.result_queue, self.match_queue),
        )
        self.match_worker_proc = mp.Process(
            target=match_worker,
            args=(self.match_queue, self.recognizer_options,
                  self.data_queue, self.output_queue),
        )
        self.static_worker_proc = mp.Process(
            target=static_worker,
            args=(self.static_queue, self.recognizer_options, self.output_queue),
        )
        for p in (self.frame_worker, self.result_worker,
                  self.match_worker_proc, self.static_worker_proc):
            p.start()

        self.current_frame_num = 0
        self.update_frame()
        self.check_output_queue()

        self.root.bind('<space>', self.next_gesture)
        self.next_gesture()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Video — thin black border gives a framed look
        video_wrap = tk.Frame(self.root, bg='#000000', padx=2, pady=2)
        video_wrap.pack(padx=10, pady=(10, 0))
        self.canvas = tk.Canvas(video_wrap, width=640, height=480,
                                bg='#000000', highlightthickness=0)
        self.canvas.pack()

        # Bottom row: instruction card (left) + log panel (right)
        bottom = tk.Frame(self.root, bg=self.BG)
        bottom.pack(fill=tk.X, padx=10, pady=8)
        bottom.columnconfigure(0, weight=3)   # instruction ~60 %
        bottom.columnconfigure(1, weight=2)   # log ~40 %

        # ── Instruction card ──────────────────────────────────────────────────
        self.instr_frame = tk.Frame(bottom, bg=self.INSTR_BG, padx=22, pady=16)
        self.instr_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))

        self._instr_sub = tk.Label(
            self.instr_frame,
            text='PERFORM THIS GESTURE',
            font=('Helvetica', 9, 'bold'),
            fg=self.INSTR_DIM, bg=self.INSTR_BG,
        )
        self._instr_sub.pack(anchor='w')

        self.gesture_label = tk.Label(
            self.instr_frame,
            text='—',
            font=('Helvetica', 30, 'bold'),
            fg=self.INSTR_FG, bg=self.INSTR_BG,
        )
        self.gesture_label.pack(anchor='w', pady=(6, 14))

        tk.Button(
            self.instr_frame,
            text='Next  →',
            font=('Helvetica', 10, 'bold'),
            fg=self.BTN_FG, bg=self.BTN_BG,
            activebackground=self.BTN_HOVER, activeforeground=self.BTN_FG,
            relief=tk.FLAT, padx=14, pady=6,
            cursor='hand2', bd=0,
            command=self.next_gesture,
        ).pack(anchor='w')

        # ── Recognition log ───────────────────────────────────────────────────
        log_frame = tk.Frame(bottom, bg=self.CARD_BG, padx=12, pady=10)
        log_frame.grid(row=0, column=1, sticky='nsew', padx=(4, 0))

        header = tk.Frame(log_frame, bg=self.CARD_BG)
        header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header,
            text='RECOGNITION LOG',
            font=('Helvetica', 9, 'bold'),
            fg=self.MUTED_FG, bg=self.CARD_BG,
        ).pack(side=tk.LEFT)

        tk.Button(
            header,
            text='Clear',
            font=('Helvetica', 9),
            fg=self.MUTED_FG, bg=self.CARD_BG,
            activebackground=self.CARD_BG,
            relief=tk.FLAT, cursor='hand2',
            command=self.clear_console,
        ).pack(side=tk.RIGHT)

        self.console = ScrolledText(
            log_frame,
            height=6,
            font=('Courier', 11),
            bg=self.CONSOLE_BG, fg=self.CONSOLE_FG,
            insertbackground=self.CONSOLE_FG,
            state='disabled', relief=tk.FLAT,
            wrap=tk.WORD, padx=6, pady=4,
        )
        self.console.pack(fill=tk.BOTH, expand=True)

    # ── Instruction card color helpers ────────────────────────────────────────
    def _set_card_colors(self, bg, fg, dim):
        self.instr_frame.config(bg=bg)
        self._instr_sub.config(bg=bg, fg=dim)
        self.gesture_label.config(bg=bg, fg=fg)

    # ── Success flash ─────────────────────────────────────────────────────────
    def _flash_success(self):
        if self._flashing:
            return
        self._flashing = True
        self._set_card_colors(self.SUCCESS_BG, self.SUCCESS_FG, self.SUCCESS_DIM)
        self._flash_job = self.root.after(1300, self._end_flash)

    def _end_flash(self):
        self._flash_job  = None
        self._flashing   = False
        self.next_gesture()

    # ── Gesture selection ─────────────────────────────────────────────────────
    def next_gesture(self, event=None):
        if self._flash_job:
            self.root.after_cancel(self._flash_job)
            self._flash_job = None
        self._flashing = False
        self._set_card_colors(self.INSTR_BG, self.INSTR_FG, self.INSTR_DIM)
        self.current_gesture_family = random.choice(self._families)
        self.gesture_label.config(text=display_name(self.current_gesture_family))

    # ── Frame loop ────────────────────────────────────────────────────────────
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.imgtk = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                point = landmarks_to_frame(results)
                data  = self.data_queue.get()
                data.append(point)
                self.data_queue.put(data)

                task = (point, self.current_frame_num, [])
                self.dynamic_queue.put(task)
                self.static_queue.put(task)
                self.current_frame_num += 1

        self.root.after(10, self.update_frame)

    # ── Recognition output ────────────────────────────────────────────────────
    def check_output_queue(self):
        while not self.output_queue.empty():
            message = self.output_queue.get()
            if not self._flashing and self._is_correct_match(message):
                self._log_success(message)
                self._flash_success()
        self.root.after(100, self.check_output_queue)

    def _is_correct_match(self, message: str) -> bool:
        if not self.current_gesture_family:
            return False
        m = re.search(r'Gesture: (\S+)', message)
        if not m:
            return False
        return gesture_family(m.group(1)) == self.current_gesture_family

    def _log_success(self, message: str):
        m = re.search(r'(Static|Dynamic) Gesture: (\S+) \| Score: (.+)', message)
        if m:
            kind  = 'STA' if m.group(1) == 'Static' else 'DYN'
            fam   = gesture_family(m.group(2))
            score = m.group(3).strip()
            line  = f'  \u2713  {display_name(fam)}  [{kind}]  score {score}\n'
        else:
            line = f'  \u2713  {message}\n'
        self.console.config(state='normal')
        self.console.insert(tk.END, line)
        self.console.see(tk.END)
        self.console.config(state='disabled')

    def _log_raw(self, message: str):
        self.console.config(state='normal')
        self.console.insert(tk.END, message + '\n')
        self.console.see(tk.END)
        self.console.config(state='disabled')

    def clear_console(self):
        self.console.config(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.config(state='disabled')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def close(self):
        for q in (self.dynamic_queue, self.static_queue,
                  self.result_queue, self.match_queue):
            q.put(None)
        self.cap.release()
        for p in (self.frame_worker, self.result_worker,
                  self.match_worker_proc, self.static_worker_proc):
            p.terminate()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app  = GestureApp(root)
    root.protocol('WM_DELETE_WINDOW', app.close)
    root.mainloop()
