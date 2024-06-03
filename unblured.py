import cv2


import argparse
import threading
import time
import numpy as np
import copy
import numpy as np
import cv2
from pathlib import Path
import copy
from sklearn.cluster import KMeans
import os
import datetime
import json

# Parser
parser = argparse.ArgumentParser(description="Process some.")
parser.add_argument("--input_video", type=str, default="./dataset/output_680.mp4")
parser.add_argument("--output_video", type=str, default="./output/counted_video.mp4")
parser.add_argument(
    "--raw_output_video", type=str, default="./output/raw_output_video.mp4"
)
parser.add_argument("--early_stop", default=-1, type=int)
parser.add_argument("--saved_every_sec",default=40, type=int)
args = parser.parse_args()

# Load video an its info.


processed_frame = None
status = "waiting"  # in two status ["waiting", "activate"]

# check if output/ folder exists, if not create it
from pathlib import Path

if not Path("output").exists():
    Path("output").mkdir(parents=True, exist_ok=True)
# Output edited video.

count = 0
save_countr = 0
prev_rails = []
prev = 100
prev_groups = None
sleeper_counter = 0
# Init State
time_format = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
folder_path = f"./output/{time_format}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def drawBlocks_binary(raw_image, size=11, sigma=5.0, theta=1, lambd=10, gamma=0.5, k1_size=5, k2_size=13):
    image = raw_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    gabor_kernel = cv2.getGaborKernel((size, size), sigma, theta, lambd, gamma, 0)
    filtered_image = cv2.filter2D(dst, -1, gabor_kernel)
    # edges = cv2.Canny(thresh, 40, 100)
    kernel = np.ones((k1_size, k1_size), np.uint8) 
    kernel_dot = np.ones((2,3), np.uint8)
    kernel_2 = np.ones(k2_size, np.uint8)

    filtered_image = cv2.erode(filtered_image, kernel_dot, iterations=1)

    filtered_image = cv2.dilate(filtered_image, kernel_2, iterations=1)
    filtered_image = cv2.erode(filtered_image, kernel, iterations=1)
    
    return filtered_image

def check_and_save(counter: int, writer: cv2.VideoWriter, name: str, frame):
    """
    check the number of frames already saved for input variable VideoWriter writer
    if writer already defined, save previous one and open new destination to be saved.
    """
    fps = 30
    total_frames = fps * args.saved_every_sec
    if counter % (total_frames) != 0:
        return writer
    if writer is not None:
        writer.release()
    writer = cv2.VideoWriter(
        f"{folder_path}/{name}_{int(counter/(total_frames))}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (len(frame[0]), len(frame)),
        isColor=True,
    )
    return writer





# Thread 1 - save raw video
def save_frames(payload):
    global save_countr
    raw_out = None
    verbose = payload.verbose
    try:

        while payload.continue_flag:
            if payload.ready_raw is False:
                payload.raw_event.clear()
                payload.raw_event.wait()
            if payload.continue_flag is False:
                break
            if verbose:
                raw_out = check_and_save(save_countr, raw_out, "raw", frame=payload.frame)
            
            frame = payload.get_raw()

            time.sleep(1e-10)
            save_countr += 1
            if verbose:
                raw_out.write(frame)
                
    except KeyboardInterrupt:
        print("ctrl+c...save_frames()")
    if raw_out:
        raw_out.release()
    print(f"saved frames: {save_countr}")


# Init processed frame as whole black
# Then update this frame as processed one.



# Thread 2 - save processed video
def save_processed(payload):
    global processed_frame
    global save_countr
    verbose = payload.verbose
    out = None
    counter = 0
    try:
        while payload.continue_flag:
            if processed_frame is None:
                payload.save_process_event.wait()
            if payload.continue_flag is False:
                break
            if verbose:
                out = check_and_save(counter, out, "out", frame=processed_frame)
            if save_countr > counter:
                
                counter += 1
                time.sleep(1e-10)
                if verbose:
                    out.write(processed_frame)

    except KeyboardInterrupt:
        print("ctrl+c...save_processed()")
    if out:
        out.release()



# Thread 3 - do process algorithm
def do_algorithm(payload):
    global processed_frame

    try:
        while payload.continue_flag:
            if payload.ready_process is False:
                payload.process_event.clear()
                payload.process_event.wait()
            frame = payload.get_process()
            processed_frame, payload.counter = algorithm(frame)
            payload.save_process_event.set()
            time.sleep(1e-10)
    except KeyboardInterrupt:
        print("ctrl+c...save_processed()")


# mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
# mask[
#     0 : int(frame_height),
#     int(frame_width / 2 - frame_height / 2) : int(frame_width / 2 + frame_height / 2),
# ] = 255

# return processed_frame, sleeper_counter
def algorithm(frame):
    global prev
    global status
    global rails_count
    global prev_groups
    global prev_rails
    global continue_flag
    global sleeper_counter
    global mask
    # Purpuse: only focus on the center features
    # Method : mask out the edge of each frame
    k1_size = 1
    k2_size = (3, 16)

    target_brightness = 135
    size = 16
    sigma = 5.5
    theta = np.pi / 2
    origin = copy.copy(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adjustment_factor = target_brightness - np.mean(gray)
    adjusted = np.clip(gray.astype("float") + adjustment_factor, 0, 255).astype("uint8")
    adjusted_bgr = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)

    frame = adjusted_bgr

    frame = cv2.resize(
        frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_CUBIC
    )
    frame = drawBlocks_binary(
        frame, size=size, sigma=sigma, theta=theta, k1_size=k1_size, k2_size=k2_size
    )
    frame = cv2.resize(
        frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_CUBIC
    )
    _, frame = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    # frame = cv2.bitwise_and(frame, frame, mask=mask)
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = np.array(
        [np.mean(np.vstack([c[:, 0][:, 0], c[:, 0][:, 1]]), axis=1) for c in contours]
    )
    try:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(
            centroids[:, 1].reshape(-1, 1)
        )
        labels = kmeans.labels_
        groups_mean = [[0, 0], [0, 0], [0, 0]]
        for label, centroid in zip(labels, centroids):
            groups_mean[label][0] += centroid[1]
            groups_mean[label][1] += 1
        ids = [mean[0] / mean[1] if mean[1] > 0 else 0 for mean in groups_mean]
    except ValueError:
        ids = []
    except IndexError:
        ids = []

    origin, sleeper_counter , status = do_conting(ids,origin, sleeper_counter, status, frame.shape[0], frame.shape[1])
    if len(frame.shape) == 2:
        # convert the grayscale image to RGB one.
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for id in ids:
        cv2.putText(
            origin,
            "i am a rail",
            (424, int(id)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    cv2.putText(origin, f"counter : {sleeper_counter}", (500, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"counter : {sleeper_counter}", (500, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # result_frame = origin
    height, width = frame.shape[0], frame.shape[1]
    bias = 0.03* height
    diff = 0.15 * height / 2
    mid = height /2
    color = (0,0,255)
    result_frame = frame
    result_frame = cv2.line(result_frame, (20, int(mid-diff)), (width-20, int(mid-diff)), color, 1)
    result_frame = cv2.line(result_frame, (20, int(mid+diff)), (width-20, int(mid+diff)), color, 1)
    concatenated_frame = cv2.hconcat([result_frame, origin])

    return concatenated_frame, sleeper_counter

def do_conting(ids,frame, sleeper_counter, status, height, width):
    bias = 0.03* height
    diff = 0.15 * height / 2
    mid = height /2
    color = (0,0,255)
    frame = cv2.line(frame, (20, int(mid-diff)), (width-20, int(mid-diff)), color, 1)
    frame = cv2.line(frame, (20, int(mid+diff)), (width-20, int(mid+diff)), color, 1)

    for id in ids:
        if status == "waiting":
            if id > mid - diff - bias and id < mid - diff + bias:
                status = "active"
                break
        else:
            if id > mid + diff - bias and id < mid + diff + bias:
                status = "waiting"
                sleeper_counter += 1
                with open(f"{folder_path}/out.json", "w") as file:
                    data = {"counting": sleeper_counter}
                    json.dump(data, file)
                break

    return frame, sleeper_counter, status


def rail_counter(payload):
    try:
        t2 = threading.Thread(target=save_frames, daemon=True, args=(payload,))
        t3 = threading.Thread(target=save_processed, daemon=True, args=(payload,))
        t4 = threading.Thread(target=do_algorithm, daemon=True, args=(payload,))

        t2.start()
        t3.start()
        t4.start()

        t2.join()
        t3.join()
        t4.join()

    except KeyboardInterrupt:
        continue_flag = False
        #t1.join()
        t2.join()
        t3.join()
        t4.join()
        print("keyboard interrupt.")

class Payload():
    def __init__(self, frame=None, verbose=True, continue_flag=True):
        self.frame = frame
        self.counter = 0
        self.verbose=verbose
        self.continue_flag=continue_flag
        self.ready_raw = False
        self.ready_process = False
        self.raw_event = threading.Event()
        self.process_event = threading.Event()
        self.save_process_event = threading.Event()

    def update(self, frame):
        self.frame = frame
        self.ready_raw = True
        self.ready_process = True
        self.raw_event.set()
        self.process_event.set()

    def get_raw(self):
        self.ready_raw = False
        return self.frame
    
    def get_process(self):
        self.ready_process = False
        return self.frame
    
    def set_finish(self):
        self.continue_flag=False
        self.raw_event.set()
        self.process_event.set()