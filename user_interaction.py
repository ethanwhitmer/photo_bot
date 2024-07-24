import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import time
from datetime import datetime
import math
import redis
import gphoto2 as gp
import threading
from pynput import keyboard
import os
#import pyrealsense 


redis_client = redis.Redis(host='localhost', port=6379, db=0)

running = True

def on_press(key):
    try:
        if key == keyboard.Key.up:
            redis_client.set('photo_bot_tele_up', 1)
            redis_client.set('photo_bot_enter_tele', 1)
        elif key == keyboard.Key.down:
            redis_client.set('photo_bot_tele_down', 1)
            redis_client.set('photo_bot_enter_tele', 1)
        elif key == keyboard.Key.left:
            redis_client.set('photo_bot_tele_left', 1)
            redis_client.set('photo_bot_enter_tele', 1)
        elif key == keyboard.Key.right:
            redis_client.set('photo_bot_tele_right', 1)
            redis_client.set('photo_bot_enter_tele', 1)
        elif key == keyboard.Key.enter:
            redis_client.set('photo_bot_enter_tele', 0)
        elif key == keyboard.Key.space:
            redis_client.set('photo_bot_photo_time', 1)
        elif key.char == "p":
            if redis_client.get('photo_bot_portrait').decode('utf-8') == '0':
                redis_client.set('photo_bot_portrait', 1)
            else:
                redis_client.set('photo_bot_portrait', 0)
        elif key.char == "a":
            redis_client.set('photo_bot_tele_ccw', 1)
            redis_client.set('photo_bot_enter_tele', 1)
        elif key.char == "d":
            redis_client.set('photo_bot_tele_cw', 1)
            redis_client.set('photo_bot_enter_tele', 1)

    except AttributeError:
        pass

def on_release(key):
    try:
        if key == keyboard.Key.up:
            redis_client.set('photo_bot_tele_up', 0)
        elif key == keyboard.Key.down:
            redis_client.set('photo_bot_tele_down', 0)
        elif key == keyboard.Key.left:
            redis_client.set('photo_bot_tele_left', 0)
        elif key == keyboard.Key.right:
            redis_client.set('photo_bot_tele_right', 0)
        elif key.char == "a":
            redis_client.set('photo_bot_tele_ccw', 0)
        elif key.char == "d":
            redis_client.set('photo_bot_tele_cw', 0)

        if key == keyboard.Key.esc:
            # Stop listener
            return False
    except AttributeError:
        pass


def arrow_key_listener():
    redis_client.set('photo_bot_tele_up', 0)
    redis_client.set('photo_bot_tele_down', 0)
    redis_client.set('photo_bot_tele_left', 0)
    redis_client.set('photo_bot_tele_right', 0)
    redis_client.set('photo_bot_tele_cw', 0)
    redis_client.set('photo_bot_tele_ccw', 0)

    redis_client.set('photo_bot_portrait', 0)
    redis_client.set('photo_bot_photo_time', 0)
    
    # Collect events until released
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

#-----------------------------------------------------------------------------

def capture_image(camera, context, capture_event):
    global target_path 
    try:
        file_path = gp.check_result(gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE))
        camera_file = gp.check_result(gp.gp_camera_file_get(camera, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL))
        target_folder = "./images"
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        target_path = os.path.join(target_folder, f"img_{current_time}.jpg")

        gp.check_result(gp.gp_file_save(camera_file, target_path))
    except gp.GPhoto2Error as ex:
        print("Error taking photo")

    capture_event.set()
    return target_path

# Function to display countdown on video feed
def display_countdown(frame, countdown, alpha):
    height, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6
    font_thickness = 20
    text = str(countdown)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    overlay = frame.copy()
    #alpha = 0.6
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


#------------------------------------------------------------------------------


def main():
    # Initialize gPhoto2 camera
    context = gp.gp_context_new()
    camera = gp.check_result(gp.gp_camera_new())
    try:
        gp.check_result(gp.gp_camera_init(camera, context))
        camera_found = True
    except gp.GPhoto2Error as ex:
        camera_found = False

    capture_flag = False
    picture_taken = False
    countdown = 3
    countdown_start_time = None
    capture_event = threading.Event()
    capture_thread = None

    # Start the arrow key listener thread
    listener_thread = threading.Thread(target=arrow_key_listener)
    listener_thread.daemon = True
    listener_thread.start()

    # Initialize the data
    rot_vec = np.array([0, 0, 0])
    trans_vec = np.array([0, 0, 0])

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        if len(result.gestures) > 0:
            top_gesture = result.gestures[0][0].category_name
            if top_gesture == 'Victory':
                redis_client.set('photo_bot_photo_time', 1)

            #print(top_gesture)

    options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='source/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

    recognizer = vision.GestureRecognizer.create_from_options(options)


    cap = cv2.VideoCapture(2)
 
    # Landmarks corresponding to forehead (54...68) and nose (1...261)
    # Features that are realtively undisturbed by facial controsion (smiling)
    landmarks_list = [54, 103, 67, 109, 10, 338, 297, 332, 284, 298, 333, 299, 337, 151, 108, 69, 104, 68, 
                    1, 4, 5, 44, 45, 51, 274, 275, 261, 
                    132, 93, 234, 127, 162,
                    288, 361, 323, 454, 356]
    face_3d = np.load('source/forward_face_3d_37pts.npy')


    window_size = 30
    face_2d_history = []
    looking_history = []

    frame_timestamp_ms = 0
       
    while running:
        success, image = cap.read()
        
        #image = image[200:500, 200:500]

        if image is not None:
            start = time.time()

            portrait_mode = redis_client.get('photo_bot_portrait').decode('utf-8') == '1'

            capture_flag = redis_client.get('photo_bot_photo_time').decode('utf-8') == '1'

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            if portrait_mode:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Take photo
            #-------------------------------------------------------------------------
            if capture_flag:
                current_time = time.time()
                if countdown_start_time is None:
                    countdown_start_time = current_time
                elapsed_time = current_time - countdown_start_time

                if elapsed_time >= 1:
                    countdown -= 1
                    countdown_start_time = current_time

                if countdown == 0 and picture_taken == False and elapsed_time < 0.5 and camera_found:
                    if (not capture_thread or not capture_thread.is_alive()):
                        capture_event.clear()
                        capture_thread = threading.Thread(target=capture_image, args=(camera, context, capture_event))
                        capture_thread.start()
                        picture_taken = True
                    

                if countdown >= 0:
                    display_countdown(image, countdown, 1-elapsed_time)

                if countdown < 0 and not camera_found:
                    redis_client.set('photo_bot_photo_time', 0)
                    countdown = 3
                    countdown_start_time = None
                    picture_taken = False
            
            #------------------------------------------------------------------------
            
            # To improve performance
            image.flags.writeable = False
            
            
            # Get the result
            results = face_mesh.process(image)
            frame_timestamp_ms += 1

            

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            recognizer.recognize_async(mp_image, frame_timestamp_ms)
           
            
            # To improve performance
            #image.flags.writeable = True
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_2d = []

            image_copy = image.copy()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in landmarks_list:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])
        
                
                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    face_2d_history.append(face_2d)
                    if len(face_2d_history) > window_size:
                        face_2d_history.pop(0)

                    face_2d_hist_array = np.array(face_2d_history, dtype=np.float64)
                    face_2d_avg = np.mean(face_2d_hist_array, axis=0)


                    # The camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                            [0, focal_length, img_h / 2],
                                            [0, 0, 1]])

                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d_avg, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    looking_history.append(rmat[:, 2])
                    if len(looking_history) > window_size:
                        looking_history.pop(0)

                    looking_history_array = np.array(looking_history, dtype=np.float64)
                    looking_avg = np.mean(looking_history_array, axis=0)

                    looking_at_vec = np.array([looking_avg[2], looking_avg[0], -looking_avg[1]])

                    looking_at_vec = looking_at_vec / np.linalg.norm(looking_at_vec)

                    # Send looking at vec to redis
                    vector_str = np.array2string(looking_at_vec, separator=',')
                    redis_client.set('looking_at_vec', vector_str)

                    # Send nose location in frame to redis
                    nose_pt = np.zeros(3)
                    nose_pt[0] = (nose_2d[0] - (img_w/2)) / (img_w/2)
                    nose_pt[1] = -(nose_2d[1] - (img_h/2)) / (img_h/2)
                    #print(nose_pt)
                    if portrait_mode:
                        nose_pt[1] = -(nose_2d[0] - (img_w/2)) / (img_w/2)
                        nose_pt[0] = (nose_2d[1] - (img_h/2)) / (img_h/2)

                    vector_str = np.array2string(nose_pt, separator=',')
                    redis_client.set('nose_pos_in_frame', vector_str)

                
                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] - looking_at_vec[1]*500), int(nose_2d[1] + looking_at_vec[2]*500) )
                    
                    cv2.line(image_copy, p1, p2, (255, 0, 0), 3)

                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime
                #print("FPS: ", fps)

                cv2.putText(image_copy, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                
                mp_drawing.draw_landmarks(
                            image=image_copy,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                

            cv2.imshow('Head Pose Estimation', image_copy)

            if capture_event.is_set():
                redis_client.set('photo_bot_photo_time', 0)
                countdown = 3
                countdown_start_time = None
                picture_taken = False
                capture_event.clear()
                cap_image = cv2.imread(target_path)
                cv2.imshow("Captured Image", cap_image)
                cv2.waitKey(1000)
                cv2.destroyWindow("Captured Image")
                
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    redis_client.set('photo_bot_enter_tele', 0)
    redis_client.set('photo_bot_tele_up', 0)
    redis_client.set('photo_bot_tele_down', 0)
    redis_client.set('photo_bot_tele_left', 0)
    redis_client.set('photo_bot_tele_right', 0)
    redis_client.set('photo_bot_tele_cw', 0)
    redis_client.set('photo_bot_tele_ccw', 0)
    redis_client.set('photo_bot_portrait', 0)
    redis_client.set('photo_bot_photo_time', 0)

    return(0)



if __name__ == "__main__":
    main()