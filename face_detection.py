import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import redis
import gphoto2 as gp
#import pyrealsense 

running = True

# Function to update the plot
def update_line(new_start, new_end, line):
    line.set_data([new_start[0], new_end[0]], [new_start[1], new_end[1]])
    line.set_3d_properties([new_start[2], new_end[2]])
    plt.draw()
    plt.pause(0.01)  # Brief pause to allow the plot to update

def on_key(event):
    global running
    if event.key == 'escape':
        running = False
        plt.close()


def main():
    r = redis.Redis(host='localhost', port=6379, db=0)

    # Initialize the data
    start_point = np.array([0, 0, 0])
    end_point = np.array([1, 1, 1])

    rot_vec = np.array([0, 0, 0])
    trans_vec = np.array([0, 0, 0])

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a line object
    line, = ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], 'b-')

    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.ion()  # Turn on interactive mode
    plt.show()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    cap = cv2.VideoCapture(0)
 
    # Landmarks corresponding to forehead (54...68) and nose (1...261)
    # Features that are realtively undisturbed by facial controsion (smiling)
    landmarks_list = [54, 103, 67, 109, 10, 338, 297, 332, 284, 298, 333, 299, 337, 151, 108, 69, 104, 68, 
                    1, 4, 5, 44, 45, 51, 274, 275, 261, 
                    132, 93, 234, 127, 162,
                    288, 361, 323, 454, 356]
    face_3d = np.load('source/forward_face_3d_37pts.npy')


    window_size = 10
    face_2d_history = []
    looking_history = []

    # Connect the key press event to the handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    while running:
        success, image = cap.read()


        if image is not None:
            start = time.time()

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Get the result
            results = face_mesh.process(image)
            
            # To improve performance
            image.flags.writeable = True
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            #face_3d = []
            face_2d = []

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

                
                    # Convert it to the NumPy array
                    #face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                            [0, focal_length, img_h / 2],
                                            [0, 0, 1]])

                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d_avg, cam_matrix, dist_matrix)

                    #rot_vec = rot_vec*1000

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
                    r.set('looking_at_vec', vector_str)

                    # Send nose location in frame to redis
                    nose_pt = np.zeros(3)
                    nose_pt[0] = (nose_2d[0] - (img_w/2)) / (img_w/2)
                    nose_pt[1] = -(nose_2d[1] - (img_h/2)) / (img_h/2)
                    #print(nose_pt)

                    vector_str = np.array2string(nose_pt, separator=',')
                    r.set('nose_pos_in_frame', vector_str)

                    
                    #print(looking_at_vec)
                    #print(" ")
                    update_line(start_point, looking_at_vec, line)

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] - looking_at_vec[1]*500), int(nose_2d[1] + looking_at_vec[2]*500) )
                    
                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime
                #print("FPS: ", fps)

                cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                
                mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                
            cv2.imshow('Head Pose Estimation', image)

    cap.release()

    return(0)



if __name__ == "__main__":
    main()