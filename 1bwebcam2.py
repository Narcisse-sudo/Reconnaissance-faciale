import os
import cv2
import time


def main() -> None:
    if os.environ.get("RF_DRY_RUN", "0") == "1":
        print("DRY_RUN: 1bwebcam2 skipped webcam access")
        return

    cam = cv2.VideoCapture(0)

    img_counter = 0
    while True:
        scd = 0.2000571966171265
        begun = time.localtime()
        begun = time.mktime(begun)
        count = scd + begun
        while True:
            ret, frame = cam.read()

            if not ret:
                print('erreur affichage')
                break
            cv2.imshow("test", frame)

            cv2.waitKey(1)

            end = time.localtime()
            end1 = time.mktime(end)
            if count >= end1:
                pass
            else:
                img_name = f"HubertineC{img_counter}.png"
                cv2.imwrite(img_name, frame)
                print("capture")
                img_counter += 1
                break

        if img_counter >= 5:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()