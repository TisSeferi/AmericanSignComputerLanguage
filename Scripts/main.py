import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import sys
import io


# Redirect stdout to Tkinter
class ConsoleRedirect(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')

    def flush(self):
        pass


class LiveFeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Feed with Gesture Recognition")

        # Video Display Canvas
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Console Output Box
        self.console = ScrolledText(root, height=8, state='normal')
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state='disabled')

        # Redirect stdout to console
        sys.stdout = ConsoleRedirect(self.console)

        # Initialize OpenCV and MediaPipe
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access the webcam.")
            return

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Start updating frames
        self.update_frame()

    def update_frame(self):
        """Capture a frame, process it with MediaPipe, and display it."""
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                print("Hand Detected")  # Example output to console
            else:
                print("No Hand Detected")  # Example output to console

            # Convert the frame to RGB for display in Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def close(self):
        """Release resources on close."""
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveFeedApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
