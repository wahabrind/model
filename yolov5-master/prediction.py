from person_detection_main import Model
import cv2


cap = cv2.VideoCapture('vid.mp4')
obj = Model()

while True:
    ret, frame = cap.read()
    dict, img = obj.run(frame)

    cv2.imshow('a', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    print(dict)
