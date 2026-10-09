
import cv2
from src.camera import Camera


def main():
    camera = Camera(camera_id=0)

    try:
        camera.open()

        while True:
            frame = camera.read()

            cv2.imshow("Hand Sign Recognition V2", frame)

            #  Q to exit
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
