import cProfile
import multiprocessing as mp
import cv2
import mediapipe as mp_solutions
import numpy as np
import os
import Jackknife as jk
from pathlib import Path
import collections as col
from Machete import Machete
from ContinuousResult import ContinuousResult, ContinuousResultOptions
from JackknifeConnector import JKConnector as jkc
from JkBlades import JkBlades

import pickle


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

def landmarks_to_frame(results):
    if results.multi_hand_landmarks:
        landmarks = [results.multi_hand_landmarks[0]]
        if landmarks:
            for handLms in landmarks:
                # Convert landmarks to dataframe
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

# Worker process for processing frames with Machete
def process_frame_worker(machete, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        point, current_frame_num, ret = task
        machete.process_frame(point, current_frame_num, ret)
        result_queue.put((ret, point, current_frame_num))

# Worker process for selecting results
def select_result_worker(result_queue, match_queue):
    while True:
        task = result_queue.get()
        if task is None:
            break
        ret, point, current_frame_num = task
        result = ContinuousResult.select_result(ret, False)
        match_queue.put((result, point, current_frame_num))

def match_worker(match_queue, recognizer_options, data_queue):
    while True:
        task = match_queue.get()
        if task is None:
            break
        result, point, current_frame_num = task
        if result:
            data = data_queue.get()

            # Look back at the previous frames starting from the end and counting back the length of the proposed gesture (given by start_frame_no-current_frame_num )
            jk_buffer = jkc.get_jk_buffer_from_video(list(data), result.start_frame_no-current_frame_num, result.end_frame_no - current_frame_num - 1)
            match, recognizer_d = recognizer_options.is_match(trajectory=jk_buffer, gid=result.sample.gesture_id)
            if match:
                print(f"Matched {result.sample.gesture_id} with score {recognizer_d}")

            data_queue.put(data)

def live_process():
    print("Starting hand detection")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Failed to open File.")
        exit()

    cr = ContinuousResultOptions()
    hands = mp_solutions.solutions.hands.Hands()
    machete = Machete(device_type=None, cr_options=cr, templates=assemble_templates())
    blades = JkBlades()
    blades.set_ip_defaults()
    
    if os.path.exists("recognizer.pkl"):
        with open("recognizer.pkl", 'rb') as f:
            recognizer_options = pickle.load(f)
    else:
        recognizer_options = jk.Jackknife(templates=assemble_templates(), blades=blades)
        with open("recognizer.pkl", 'wb') as f:
            pickle.dump(recognizer_options, f)
    
    # Setup multiprocessing queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    match_queue = mp.Queue()
    data_queue = mp.Queue()

    # Put initial empty deque into the data_queue. Set max size to 100 for testing.
    # Everything that refers to the data_queue will be counting back from the end of the queue.
    data_queue.put(col.deque(maxlen=100))

    # Setup worker processes
    frame_worker = mp.Process(target=process_frame_worker, args=(machete, task_queue, result_queue))
    result_worker = mp.Process(target=select_result_worker, args=(result_queue, match_queue))
    match_worker_proc = mp.Process(target=match_worker, args=(match_queue, recognizer_options, data_queue))

    # Start worker processes
    frame_worker.start()
    result_worker.start()
    match_worker_proc.start()

    current_frame_num = 0
    while True:
        success, img = cap.read()
        if not success:
            print("Error in reading!")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            point = landmarks_to_frame(results)

            # Get the current deque from the data_queue, append the point, and put it back
            data = data_queue.get()
            data.append(point)
            data_queue.put(data)

            task_queue.put((point, current_frame_num, []))
            current_frame_num += 1

        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Terminate processes
    task_queue.put(None)
    result_queue.put(None)
    match_queue.put(None)
    
    frame_worker.join()
    result_worker.join()
    match_worker_proc.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    def run_profile():
        live_process()

    cProfile.run('run_profile()')
