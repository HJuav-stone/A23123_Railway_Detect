
# import Timber's file
import examine as rail_counter
from unblured import Payload

import random
import cv2
import threading
import time
import os
def timbers_code(payload):
    rail_counter.rail_counter(payload=payload)



continue_flag = True
def main():
    # 初始化一個thread 跟存放共用資料的payload
    # verbose 決定是否要存下影片    
    payload = Payload(frame=None, continue_flag=True, verbose=True)
    t1 = threading.Thread(target=timbers_code, daemon=True, args=(payload, ))
    
    t1.start()
    folder_path = "/Users/zhengtingwei/Datasets/UAV/20240510/20240510030149.mp4"
    cap = cv2.VideoCapture(folder_path) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        payload.update(frame)
        time.sleep(1/120)
    print("main done.")
    # 結束後傳送下述的flag讓thread知道要結束了
    payload.set_finish()
    t1.join()

    # 從payload.counter取得值
    print(payload.counter)

main()
