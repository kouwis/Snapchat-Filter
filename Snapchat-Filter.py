import cv2 as cv
import mediapipe as mp


def ImageProc(img):
    img2 = cv.imread("heart.png")
    imgHead = cv.imread("Head.png")

    img2g = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    imgHg = cv.cvtColor(imgHead, cv.COLOR_BGR2GRAY)

    _, MaskEye = cv.threshold(img2g, 50, 255, cv.THRESH_BINARY_INV)
    _, MaskHead = cv.threshold(imgHg, 25, 255, cv.THRESH_BINARY_INV)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img2, imgHead, MaskEye, MaskHead, imgRGB


def RES(img, listt, results):
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for id, lm in enumerate(face.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                listt.append([id, cx, cy])
    return listt


def LeftEye(img, img2, MaskEye, listt):
    LeftEyeXL, LeftEyeYL = listt[33][1], listt[33][2]
    LeftEyeXR, LeftEyeYR = listt[133][1], listt[133][2]
    LeftEyeXU, LeftEyeYU = listt[159][1], listt[159][2]
    LeftEyeXD, LeftEyeYD = listt[145][1], listt[145][2]

    img2L = cv.resize(img2, ((LeftEyeXR - LeftEyeXL) + 14, (LeftEyeYD - LeftEyeYU) + 14))
    left_eye = img[LeftEyeYU - 7: LeftEyeYD + 7, LeftEyeXL - 7:LeftEyeXR + 7]
    BitL = cv.bitwise_and(left_eye, left_eye, MaskEye)
    finialL = cv.add(BitL, img2L)
    img[LeftEyeYU - 7:LeftEyeYD + 7, LeftEyeXL - 7:LeftEyeXR + 7] = finialL

    return img


def RightEye(img, img2, MaskEye, listt):
    RightEyeXL, RightEyeYL = listt[362][1], listt[362][2]
    RightEyeXR, RightEyeYR = listt[263][1], listt[263][2]
    RightEyeXU, RightEyeYU = listt[386][1], listt[386][2]
    RightEyeXD, RightEyeYD = listt[374][1], listt[374][2]

    img2R = cv.resize(img2, ((RightEyeXR - RightEyeXL) + 14, (RightEyeYD - RightEyeYU) + 14))
    right_eye = img[RightEyeYU - 7: RightEyeYD + 7, RightEyeXL - 7:RightEyeXR + 7]
    BitR = cv.bitwise_and(right_eye, right_eye, MaskEye)
    finialR = cv.add(BitR, img2R)
    img[RightEyeYU - 7:RightEyeYD + 7, RightEyeXL - 7:RightEyeXR + 7] = finialR

    return img


def Head(img, imgHead, MaskHead, listt):
    HeadXL, HeadYL = listt[104][1], listt[104][2]
    HeadXR, HeadYR = listt[298][1], listt[298][2]
    HeadXU, HeadYU = listt[338][1], listt[338][2]
    HeadXD, HeadYD = listt[337][1], listt[337][2]

    imgH = cv.resize(imgHead, ((HeadXR - HeadXL) + 30, (HeadYD - HeadYU) + 50))
    head = img[HeadYU - 40: HeadYD + 10, HeadXL - 15:HeadXR + 15]
    BitH = cv.bitwise_and(head, head, MaskHead)
    finialH = cv.add(BitH, imgH)
    img[HeadYU - 40: HeadYD + 10, HeadXL - 15:HeadXR + 15] = finialH

    return img


def main():
    cap = cv.VideoCapture(0)

    w, h = 420, 360
    cap.set(3, w)
    cap.set(4, h)

    mpFace = mp.solutions.face_mesh
    Face = mpFace.FaceMesh()

    while 1:

        _, img = cap.read()
        img2, imgHead, MaskEye, MaskHead, imgRGB = ImageProc(img)
        results = Face.process(imgRGB)

        listt = []
        listt = RES(img, listt, results)

        if len(listt) == 0:
            continue

        img = LeftEye(img, img2, MaskEye, listt)
        img = RightEye(img, img2, MaskEye, listt)
        img = Head(img, imgHead, MaskHead, listt)

        cv.imshow("SnapChat Filter", img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
