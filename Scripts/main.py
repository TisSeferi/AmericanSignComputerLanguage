import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import threading

class GestureSandbox:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Sandbox")

        # Video feed display
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Buttons
        tk.Button(root, text="Start Recording", command=self.start_recording).pack()
        tk.Button(root, text="Stop Recording", command=self.stop_recording).pack()
        tk.Button(root, text="Play Gesture", command=self.play_gesture).pack()
        tk.Button(root, text="Save Gesture", command=self.save_gesture).pack()
        tk.Button(root, text="Load Gesture", command=self.load_gesture).pack()

        # State variables
        self.recording = False
        self.gesture_data = []
        self.current_gesture = None
        self.current_frame = 0
        self.paused = False
        self.ax = None  # Store Matplotlib axis

        # MediaPipe setup
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        # Start updating the GUI with video feed
        self.update_frame()

    def update_frame(self):
        """Capture video frames and display them in the Tkinter canvas."""
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for MediaPipe and Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    if self.recording:
                        # Record landmarks if in recording mode
                        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        self.gesture_data.append(landmarks)

            # Convert frame for display in Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def start_recording(self):
        """Start recording gesture data."""
        self.recording = True
        self.gesture_data = []
        print("Recording started...")

    def stop_recording(self):
        """Stop recording gesture data."""
        self.recording = False
        self.current_gesture = np.array(self.gesture_data)
        print(f"Recording stopped. {len(self.gesture_data)} frames captured.")

    def play_gesture(self):
        """Play back the recorded gesture with frame-by-frame control."""
        if self.current_gesture is None:
            print("No gesture recorded.")
            return

        self.current_frame = 0
        self.paused = False

        def on_key(event):
            """Handle keyboard input for playback control."""
            if event.key == ' ':
                # Toggle pause
                self.paused = not self.paused
            elif event.key == 'right':
                # Next frame
                self.current_frame = min(self.current_frame + 1, len(self.current_gesture) - 1)
                self.update_plot()
            elif event.key == 'left':
                # Previous frame
                self.current_frame = max(self.current_frame - 1, 0)
                self.update_plot()

        def playback():
            """Handle the playback loop."""
            plt.ion()  # Interactive mode
            fig, self.ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', on_key)

            while self.current_frame < len(self.current_gesture):
                if not self.paused:
                    self.update_plot()
                    self.current_frame += 1
                plt.pause(0.1)

            plt.ioff()  # Turn off interactive mode
            plt.close()

        threading.Thread(target=playback).start()

    def update_plot(self):
        """Update the plot with the current frame."""
        if self.ax is None:
            return
        frame = self.current_gesture[self.current_frame]
        self.ax.clear()
        self.ax.scatter([lm[0] for lm in frame], [lm[1] for lm in frame])
        self.ax.set_title(f"Frame {self.current_frame + 1}/{len(self.current_gesture)}")
        plt.draw()

    def save_gesture(self):
        """Save the recorded gesture to a file."""
        if self.current_gesture is None:
            print("No gesture to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy Files", "*.npy")])
        if file_path:
            np.save(file_path, self.current_gesture)
            print(f"Gesture saved to {file_path}.")

    def load_gesture(self):
        """Load a gesture from a file."""
        file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
        if file_path:
            self.current_gesture = np.load(file_path)
            print(f"Gesture loaded from {file_path}.")

    def close(self):
        """Release resources on close."""
        self.cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close any OpenCV windows
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureSandbox(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()