import cv2
import numpy as np
import pafy

#cuda 사용가능여부 확인
#ount = cv2.cuda.getCudaEnabledDeviceCount()
#print(count)

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

# YOLO 네트워크 불러오기
net = cv2.dnn.readNet("C:\\Users\\admin\\PycharmProjects\\peoplecounting\\yolov3\\yolov3.weights",
                      "C:\\Users\\admin\\PycharmProjects\\peoplecounting\\yolov3\\yolov3.cfg")

#클래스의 개수만큼 랜덤 RGB 배열 생성
colors = np.random.uniform(0, 255, size=(len(classes),3))

def yolo(frame,size, score_threshold, nms_threshold):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    #네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size,size), (0,0,0), True, crop=False)

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
            center_x = (x+w)/2
            center_y = (y+h)/2
            centers = np.zeros(shape=(len(boxes),2))
            centers[i] = [center_x,center_y]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

            # 탐지된 객체의 정보 출력
            print(f"[{class_name}({i})] conf: {confidences[i]} / x:{x} y:{y} width:{w} height:{h} / centers:{centers[i]} ")
    return frame

url = "https://www.youtube.com/watch?v=DW8x_EqZnxU"
video = pafy.new(url)
print("video title : {}".format(video.title))  # 제목
print("video duration : {}".format(video.duration))  # 길이
best = video.getbest(preftype="mp4")
print("best resolution : {}".format(best.resolution))

capture = cv2.VideoCapture(best.url)
#capture = capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture("rtsp://admin:admin123!@192.168.0.51/profile2/media.smp")
#capture = cv2.VideoCapture("C:\\Users\\admin\\Desktop\\test.avi")

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]

while True:
    ret, frame = capture.read() #프레임 읽기:카메라의 상태, 프레임
    #frame2 = cv2.resize(frame,dsize=(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    frame3 = yolo(frame, size=size_list[0], score_threshold=0.4, nms_threshold=0.4)

    cv2.imshow("VideoFrame", frame3) #이미지 표시(윈도우 창제목, 이미지)
    cv2.waitKey(1) # 33밀리초동안, 키 입력이 있을때까지 프로그램을 지연시킴
capture.release()
cv2.destroyAllWindows()