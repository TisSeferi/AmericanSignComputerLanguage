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
        self.canvas.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Start Recording", command=self.start_recording).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Stop Recording", command=self.stop_recording).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Play Gesture", command=self.play_gesture).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Gesture", command=self.save_gesture).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Load Gesture", command=self.load_gesture).pack(side=tk.LEFT, padx=5)

        # Controls info
        controls_text = """
        Playback Controls:
        Space - Play/Pause
        Left Arrow - Previous Frame
        Right Arrow - Next Frame
        E - Mark Current Frame as Start
        R - Mark Current Frame as End
        """
        tk.Label(root, text=controls_text, justify=tk.LEFT).pack(pady=10)

        # State variables
        self.recording = False
        self.gesture_data = []
        self.current_gesture = None
        self.current_frame = 0
        self.paused = False
        self.ax = None
        self.start_frame = None
        self.end_frame = None

        # MediaPipe setup
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        self.connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 17), (0, 13), (0, 9), (0, 5)
        ]

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
                self.paused = not self.paused
            elif event.key == 'right':
                self.current_frame = min(self.current_frame + 1, len(self.current_gesture) - 1)
                self.update_plot()
            elif event.key == 'left':
                self.current_frame = max(self.current_frame - 1, 0)
                self.update_plot()
            elif event.key == 'e':
                self.start_frame = self.current_frame
                print(f"Start frame set to: {self.start_frame}")
                self.update_plot()
            elif event.key == 'r':
                self.end_frame = self.current_frame
                print(f"End frame set to: {self.end_frame}")
                self.update_plot()

        def playback():
            """Handle the playback loop."""
            plt.ion()  # Interactive mode
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')
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
        
        # Create 3D scatter plot
        xs = [lm[0] for lm in frame]
        ys = [lm[1] for lm in frame]
        zs = [lm[2] for lm in frame]
        
        # Plot landmarks
        self.ax.scatter(xs, ys, zs, c='blue', s=20)
        
        # Draw connections
        for connection in self.connections:
            start_idx, end_idx = connection
            self.ax.plot([xs[start_idx], xs[end_idx]], 
                        [ys[start_idx], ys[end_idx]], 
                        [zs[start_idx], zs[end_idx]], 
                        c='red', linewidth=1)
        
        # Set fixed axes limits
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_zlim([-0.1, 0.1])
        
        # Add labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        title = f"Frame {self.current_frame + 1}/{len(self.current_gesture)}"
        if self.start_frame is not None:
            title += f" (Start: {self.start_frame})"
        if self.end_frame is not None:
            title += f" (End: {self.end_frame})"
        self.ax.set_title(title)
        
        plt.draw()

    def save_gesture(self):
        """Save the recorded gesture to a file."""
        if self.current_gesture is None:
            print("No gesture to save.")
            return

        # If start and end frames are set, only save that segment
        if self.start_frame is not None and self.end_frame is not None:
            gesture_to_save = self.current_gesture[self.start_frame:self.end_frame + 1]
        else:
            gesture_to_save = self.current_gesture

        file_path = filedialog.asksaveasfilename(defaultextension=".npy", 
                                               filetypes=[("Numpy Files", "*.npy")])
        if file_path:
            np.save(file_path, gesture_to_save)
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