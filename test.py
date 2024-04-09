from mediapipe import solutions #Importing mediapipe
from mediapipe.framework.formats import landmark_pb2 #Importing mediapipe
import numpy as np #Importing numpy for image
import math #Importing math to calculate angle

def draw_landmarks_on_image(rgb_image, detection_result): #Drawing the landmarks on image

  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)): #Running through list of landmarks
    pose_landmarks = pose_landmarks_list[idx] #Picking selected landmarks
    
    pose_12x=pose_landmarks_list[0][12].x
    pose_12y=pose_landmarks_list[0][12].y
    
    pose_14x=pose_landmarks_list[0][14].x
    pose_14y=pose_landmarks_list[0][14].y
    
    pose_16x=pose_landmarks_list[0][16].x
    pose_16y=pose_landmarks_list[0][16].y
   
    pose_24x=pose_landmarks_list[0][24].x
    pose_24y=pose_landmarks_list[0][24].y
    
    pose_26x=pose_landmarks_list[0][26].x
    pose_26y=pose_landmarks_list[0][26].y
   
    pose_30x=pose_landmarks_list[0][30].x
    pose_30y=pose_landmarks_list[0][30].y
    
    def calculate_angle(x24,y24,x26,y26,x30,y30): #Function to calculate angle
          
      ab=(y26-y24)/(x26-x24)
      bc=(y30-y26)/(x30-x26)
      
      angle = math.degrees(math.atan((bc - ab) / (1 + ab * bc)))
    
      return angle
    
    angle = calculate_angle(pose_24x, pose_24y, pose_26x, pose_26y, pose_30x, pose_30y) #Angle for knee bend
    angle+=180
    print(f"Angle: {angle}")
    # Desired angle: 154.53

    if angle>=150 and angle<=160:
          print("Good knee bend")
    elif angle<150:
          print("Bend knee less")
    elif angle>160:
          print("Bend knee more")
    else:
          print("Invalid")
    
    angle2 = calculate_angle(pose_12x, pose_12y, pose_24x, pose_24y, pose_26x, pose_26y) #Angle of back bend
    angle2=180-angle2
    print(f"Angle: {angle2}")
    
    if angle2>=135 and angle2<=145:
          print("Good back bend")
    elif angle2<135:
          print("Bend back less")
    elif angle2>145:
          print("Bend back more")
    else:
          print("Invalid")
          
    angle3 = calculate_angle(pose_12x, pose_12y, pose_14x, pose_14y, pose_16x, pose_16y) #Angle for elbow bend
    angle3=180-angle3
    print(f"Angle: {angle3}")
    
    if angle3>=165 and angle3<=175:
          print("Good elbow bend")
    elif angle3<165:
          print("Bend elbow less")
    elif angle3>175:
          print("Bend elbow more")
    else:
          print("Invalid")
         
    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y) for landmark in pose_landmarks
      #landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(base_options=base_options,output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("Bad_Posture.jpeg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)
#print(detection_result)

import cv2

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imwrite("testing_image.jpg",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imwrite("mask.jpg",visualized_mask)