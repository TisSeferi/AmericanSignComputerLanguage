import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import cv2
import numpy as np
import multiprocessing as mp
from pathlib import Path
import collections as col
from Machete import Machete
from ContinuousResult import ContinuousResult, ContinuousResultOptions
from JackknifeConnector import JKConnector as jkc
from JkBlades import JkBlades
import Jackknife as jk
import mathematics
from Vector import Vector
import pickle
from PIL import Image, ImageTk
import os
import mediapipe as mp_solutions
import time
import math
import random

# Constants
X = 0
Y = 1
Z = 2

NUM_POINTS = 21
DIMS = 3
CV2_RESIZE = (640, 480)
RAW_VIDS_FOLDER = 'templatevids/'
HOME = str(Path(__file__).resolve().parent.parent)
TEMPLATES = HOME + '\\templates\\'
TESTS = HOME + '\\testvideos\\'

# Utility function to convert landmarks to frame data
def landmarks_to_frame(results):
    if results.multi_hand_landmarks:
        landmarks = [results.multi_hand_landmarks[0]]
        if landmarks:
            for handLms in landmarks:
                points = handLms.landmark
                frame = np.zeros((NUM_POINTS * DIMS))
                for ii, lm in enumerate(points):
                    ii = ii * 3
                    frame[ii + X] = lm.x
                    frame[ii + Y] = lm.y
                    frame[ii + Z] = lm.z
        return frame

# Load templates
def assemble_templates():
    templates = []
    for path in os.listdir(TEMPLATES):
        name = path.split('.')[0]
        templates.append((name, np.load(TEMPLATES + '/' + path)))
    return templates

# Worker for frame processing
def process_frame_worker(machete, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        point, current_frame_num, ret = task
        machete.process_frame(point, current_frame_num, ret)
        result_queue.put((ret, point, current_frame_num))

# Worker for selecting results
def select_result_worker(result_queue, match_queue):
    while True:
        task = result_queue.get()
        if task is None:
            break
        ret, point, current_frame_num = task
        result = ContinuousResult.select_result(ret, False)
        match_queue.put((result, point, current_frame_num))

# Worker for gesture matching
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
                    output_queue.put(f"Dynamic Gesture: {result.sample.gesture_id} | Score: {recognizer_d:.2f}")
            finally:
                data_queue.put(data)

def static_worker(task_queue, recognizer_options, output_queue):
    match_history = col.deque(maxlen=3)
    while True:
        task = task_queue.get()
        if task is None:
            break
        point, current_frame_num, ret = task        
        point_vec = Vector(point)
        point_centroid = mathematics.calculate_centroid(point_vec)
        point_vecs_flat, _ = mathematics.convert_joint_positions_to_distance_vectors(point_vec, point_centroid)
        best_match = None
        best_distance = float('-inf')

        # Only consider static templates
        static_templates = [t for t in recognizer_options.templates if t.features.is_static]
        for template in static_templates:
            ff_vec_distance_list, total_distance = mathematics.calculate_joint_angle_disparity(
                template.features.ff_joint_vecs_flat,
                point_vecs_flat
            )
            threshold = 0.9 * NUM_POINTS  # Adjust the multiplier as needed
            if total_distance > best_distance and total_distance > threshold: 
                best_distance = total_distance
                best_match = template
                                       
        if best_match:
            match_history.append(best_match.gesture_id)
            #movement_ratio = best_match.features.path_length / best_match.features.ff_bb_magnitude
            
            if len(match_history) == 3 and len(set(match_history)) == 1:
                #debug_info = (
                #    f"Consensus Gesture: {best_match.gesture_id}\n"
                #    f"Score: {best_distance:.2f}\n"
                #    f"Path Length: {best_match.features.path_length:.2f}\n"
                #    f"Movement Ratio: {movement_ratio:.2f}\n"
                #    f"-------------------"
                #)
                output_queue.put(f"Static Gesture: {best_match.sample.gesture_id} | Score: {best_distance:.2f}")
                match_history.clear()  # Clear history after a match

# GUI Application
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.last_log_time = 0
        self.root.title("Gesture Recognition GUI")

        # Canvas for video display
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Console output box
        self.console = ScrolledText(root, height=8, state='normal', font=('Arial', 16, 'bold'))
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state='disabled')

        # Add a clear button to your GUI:
        self.clear_button = tk.Button(root, text="Clear Console", command=self.clear_console)
        self.clear_button.pack()

        # Add a textbox for displaying the template name
        self.template_display = tk.Text(root, height=1, font=('Arial', 16, 'bold'), state='disabled')
        self.template_display.pack()

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("Error: Cannot access the webcam.")
            return
        
        self.hands = mp_solutions.solutions.hands.Hands()

        # Setup queues
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.match_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.data_queue = mp.Queue()
        self.data_queue.put(col.deque(maxlen=100))

        # Load templates and initialize workers
        self.machete = Machete(device_type=None, cr_options=ContinuousResultOptions(), templates=assemble_templates())
        self.templates = [t[0] for t in assemble_templates()]
        self.current_template = None

        self.blades = JkBlades()
        self.blades.set_ip_defaults()

        if os.path.exists("recognizer.pkl"):
            with open("recognizer.pkl", 'rb') as f:
                self.recognizer_options = pickle.load(f)
        else:
            self.recognizer_options = jk.Jackknife(templates=assemble_templates(), blades=self.blades)
            with open("recognizer.pkl", 'wb') as f:
                pickle.dump(self.recognizer_options, f)

        # Start worker processes
        self.frame_worker = mp.Process(target=process_frame_worker, args=(self.machete, self.task_queue, self.result_queue))
        self.result_worker = mp.Process(target=select_result_worker, args=(self.result_queue, self.match_queue))
        self.match_worker_proc = mp.Process(target=match_worker, args=(self.match_queue, self.recognizer_options, self.data_queue, self.output_queue))
        self.static_worker = mp.Process(target=static_worker, args=(self.task_queue, self.recognizer_options, self.output_queue))

        self.frame_worker.start()
        self.result_worker.start()
        self.match_worker_proc.start()
        self.static_worker.start()

        # Start GUI updates
        self.current_frame_num = 0
        self.update_frame()
        self.check_output_queue()

        self.root.bind("<space>", self.change_template)
        self.change_template()

    def update_frame(self):
        """Update video frame in GUI."""
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.imgtk = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                point = landmarks_to_frame(results)
                data = self.data_queue.get()
                data.append(point)
                self.data_queue.put(data)

                self.task_queue.put((point, self.current_frame_num, []))
                self.current_frame_num += 1

        self.root.after(10, self.update_frame)

    def check_output_queue(self):
        """Poll output queue for results."""
        while not self.output_queue.empty():
            message = self.output_queue.get()
            self.log(message)

        self.root.after(100, self.check_output_queue)

    def log(self, message):
        """Log a message to the console box with rate limiting."""
        current_time = time.time()
        # Only log if 0.8 seconds have passed since last message
        if current_time - self.last_log_time >= 0.7:  
            self.console.config(state='normal')
            self.console.insert(tk.END, f"{message}\n")
            self.console.see(tk.END)
            self.console.config(state='disabled')
            self.last_log_time = current_time

    def clear_console(self):
        self.console.config(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.config(state='disabled')

    def change_template(self, event=None):
        """Change the displayed template to a random one."""
        if self.templates:
            self.current_template = random.choice(self.templates)
            self.template_display.config(state='normal')
            self.template_display.delete(1.0, tk.END)
            self.template_display.insert(tk.END, f"Do This: {self.current_template}")
            self.template_display.config(state='disabled')

    def close(self):
        """Clean up resources."""
        self.task_queue.put(None)
        self.result_queue.put(None)
        self.match_queue.put(None)
        self.cap.release()
        self.frame_worker.terminate()
        self.result_worker.terminate()
        self.match_worker_proc.terminate()
        self.static_worker.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
