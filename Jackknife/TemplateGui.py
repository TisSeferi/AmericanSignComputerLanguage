import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import cv2
from PIL import Image, ImageTk

def append_to_console(message):
    console_output.configure(state='normal')
    console_output.insert(tk.END, message + '\n')  
    console_output.configure(state='disabled')  
    console_output.see(tk.END)

def update_frame():
    ret, cv_image = cap.read()
    if ret:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 
        pil_image = Image.fromarray(cv_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.image = tk_image 
        canvas.itemconfig(image_on_canvas, image=tk_image)
    main.after(10, update_frame) 

main = tk.Tk()
main.title('ASCL Prototype')
main.geometry('1200x700')

title_label = ttk.Label(main, text='ASCL', font='Calibri 24 bold')
title_label.pack(pady=10)

canvas = tk.Canvas(main, width=400, height=400)
canvas.pack(side=tk.LEFT, padx=20, pady=10)
image_on_canvas = canvas.create_image(200, 200, anchor='center')

record_frame = ttk.Frame(main)
record_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

record_button = ttk.Button(record_frame, text='Record')
record_button.pack(side=tk.RIGHT, padx=10)

recordP = ttk.Entry(record_frame)
recordP.pack(side=tk.RIGHT, padx=10)

run_frame = ttk.Frame(main)
run_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

run_button = ttk.Button(run_frame, text='Run')
run_button.pack(side=tk.RIGHT, padx=10)

runP = ttk.Entry(run_frame)
runP.pack(side=tk.RIGHT, padx=10)

live_frame = ttk.Frame(main)
live_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

live_button = ttk.Button(live_frame, text='Live')
live_button.pack(side=tk.RIGHT, padx=10)

console_output = scrolledtext.ScrolledText(main, height=16)
console_output.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 20))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    append_to_console("Failed to open camera.")
    raise ValueError("Unable to open video source")

update_frame()

main.mainloop()

cap.release()
cv2.destroyAllWindows()
