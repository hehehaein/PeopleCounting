import cv2
import numpy as np
import pafy
import argparse
import sys
import os.path
import math
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject


# cuda 사용가능여부 확인
# ount = cv2.cuda.getCudaEnabledDeviceCount()
# print(count)


# 클래스 리스트
classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

#인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
#입력받을 인자값 등록
parser.add_argument('--video', help='Path to video file.')
#입력받은 인자값을 arg에 저장 (type:namespace)
args = parser.parse_args()

# initialize the vide
# o writer
writer = None
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# YOLO 네트워크 불러오기
net = cv2.dnn.readNet("C:\\Users\\admin\\PycharmProjects\\peoplecounting\\yolov3\\yolov3.weights",
                      "C:\\Users\\admin\\PycharmProjects\\peoplecounting\\yolov3\\yolov3.cfg")


url = "https://www.youtube.com/watch?v=DW8x_EqZnxU"
video = pafy.new(url)
print("video title : {}".format(video.title))  # 제목
print("video duration : {}".format(video.duration))  # 길이
best = video.getbest(preftype="mp4")
print("best resolution : {}".format(best.resolution))


# 클래스의 개수만큼 랜덤 RGB 배열 생성
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def yolo(frame, size, score_threshold, nms_threshold):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []
    centers = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)

    # 후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')
    print("\n\n============================== classes ==============================")

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            center_x = (x + w) / 2
            center_y = (y + h) / 2
            centers = np.zeros(shape=(len(boxes), 2))
            centers[i] = [center_x, center_y]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            # 탐지된 객체의 정보 출력
            print(
                f"[{class_name}({i})] conf: {confidences[i]} / x:{x} y:{y} width:{w} height:{h} / centers:{centers[i]} ")
    return frame


# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    # Draw a center of a bounding box
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    cv2.line(frame, (0, frameHeight // 2 - 50), (frameWidth, frameHeight // 2 - 50), (0, 255, 255), 2)
    cv2.circle(frame, (left + (right - left) // 2, top + (bottom - top) // 2), 3, (0, 0, 255), -1)

    counter = []
    if (top + (bottom - top) // 2 in range(frameHeight // 2 - 2, frameHeight // 2 + 2)):
        coun = 0
        coun += 1
        # print(coun)

        counter.append(coun)

    label = 'Pedestrians: '.format(str(counter))
    cv2.putText(frame, label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # 네트워크에서 출력되는 모든 경계 상자를 스캔하고 다음 항목만 유지합니다.
    # 자신감 점수가 높은 사람들. 상자의 클래스 레이블을 점수가 가장 높은 클래스로 지정합니다.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # non maximum suppression를 수행하여 중복된 상자 제거
    # 신뢰도 낮아짐.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)
            counting(objects)

            # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                if direction < 0 and centroid[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] in range(frameHeight // 2 - 10, frameHeight // 2 + 10):
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (50, frameHeight - ((i * 20) + 50)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
# 네트워크에 넣기 위한 전처리

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

if (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(0)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv2.VideoCapture("C:\\Users\\admin\\Desktop\\test.avi")

# Get the video writer initialized to save the output video
#vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                            #(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cv2.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    cv2.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes

    #vid_writer.write(frame.astype(np.uint8))

    cv2.imshow(winName, frame)








"""capture = cv2.VideoCapture(best.url)
# capture = capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture("rtsp://admin:admin123!@192.168.0.51/profile2/media.smp")
# capture = cv2.VideoCapture("C:\\Users\\admin\\Desktop\\test.avi")

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]

while True:
    ret, frame = capture.read()  # 프레임 읽기:카메라의 상태, 프레임
    # frame2 = cv2.resize(frame,dsize=(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    frame3 = yolo(frame, size=size_list[0], score_threshold=0.4, nms_threshold=0.4)

    cv2.imshow("VideoFrame", frame3)  # 이미지 표시(윈도우 창제목, 이미지)
    cv2.waitKey(1)  # 33밀리초동안, 키 입력이 있을때까지 프로그램을 지연시킴
capture.release()
cv2.destroyAllWindows()"""
