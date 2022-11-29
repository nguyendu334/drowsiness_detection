from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import winsound

start_time = time.time()

def ty_so_mat(eye):
# tinh khoang cach euclide giua 2 bo danh dau mat doc toa do (x, y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
# tinh khoang cach euclide giua diem moc danh dau mat ngang toa do (x, y)
    C = dist.euclidean(eye[0], eye[3])
# tinh ti le mat
    ear = (A + B) / (2.0 * C)
# tra ve ti le mat
    return ear

# xac dinh 2 hang so, 1 cho ti le khia canh mat de biet mat nhap nhay va 1 cho so khung lien tiep ma mat phai nam duoi nguong de dat canh bao
NGUONG_MAT = 0.25
SO_KHUNG_HINH = 5
DEM = 0

# khoi tao bo phat hien khuon mat cua opencv (dua tren haar cascade) va tao ra bo du doan danh dau moc cua khuon mat cua dlib
#print("[INFO] loading facial landmark predictor...")

detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "68_face_landmarks_predictor.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# lay cac chi so cua cac dau moc tren khuon mat cho mat trai va mat phai tuong ung
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# bat dau thu luong video
print("[INFO] starting video stream thread...")
vs = VideoStream(0).start()
time.sleep(1.0)
print ( "--- %s seconds ---" % (time.time() - start_time))
# vong lap qua cac khung hinh tu luong video
while True:
    start_time_1 = time.time()
    # lay khung tu luong tep video, thay doi kich thuoc va chuyen doi no sang cac kenh mau xam
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # phat hien cac khuon mat trong khung anh xam
    rects = detector(gray, 0)

# vong lap tren phat hien khuon mat
    for rect in rects:
        # xac dinh diem moc tren mat doi voi vung khuon mat, sau do chuyen doi diem moc tren mat toa do (x, y) voi mangNumPy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # trich xuat toa do mat trai va phai, sau do su dung toa do de tinh ti le mat cho ca2 mat
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = ty_so_mat(leftEye)
        rightEAR = ty_so_mat(rightEye)
        # tinh ti le mat trung binh cho ca 2 mat
        ear = (leftEAR + rightEAR) / 2.0
    # tinh vien bao loi cho mat trai va mat phai, sau do hinh dung ra (ve ra) vien bao do cho moi mat
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    # kiem tra ti le mat co nam duoi nguong nhay mat hay khong, neu co tang bo dem khung nhap nhay
        if ear < NGUONG_MAT: 
            DEM += 1
        # neu nham mat du so luong khung da dat thi bao dong
            if DEM > SO_KHUNG_HINH: 
                cv2.putText(frame, "Sleep detected! Alarm!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                        lineType=cv2.LINE_AA)
                # winsound.Beep(400, 500)
        # mat khac neu ty le mat khong duoi nguong nhay mat
        else: 
            DEM = 0
        # ve thong so ty le mat da tinh tren khung de giup viec kiem tra sua loi va thiet lap lai dung nguong ty le mat va bo dem khung
        cv2.putText(frame, "TY SO MAT: {:.2f}".format(ear), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # show khung hinh
    cv2.imshow("PHAT HIEN NGU GAT", frame)
    key = cv2.waitKey(1) & 0xFF
    # nhan 'q' de thoat khoi vong lap va xoa ngo ra pi
    if key == ord("q"):
        break
    print ( "--- %s seconds ---" % (time.time() - start_time_1))
    # ngung thu video va dong tat ca cua so
cv2.destroyAllWindows()
vs.stop()