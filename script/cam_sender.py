# run this script on local computer attached camera device

from numpysocket import NumpySocket
import cv2
import argparse


def main():
    cap = cv2.VideoCapture(0)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    thresh = read_fps / fps
    frame_counter = 0

    np_sock = NumpySocket()
    np_sock.startClient(host_ip, port)

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter += 1

        if ret is False:
            break
        if frame_counter >= thresh:
            try:
                frame_resize = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
                np_sock.send(frame_resize)
            except TypeError:
                break
            frame_counter = 0
    np_sock.close()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='''%(prog)s [-h] ''',
        description='Camera image sender.'
    )
    parser.add_argument(
        "--host_ip",
        type=str,
        required=True,
        help="server's ip")
    parser.add_argument(
        "--port",
        type=int,
        default=41234,
        help="server's port number")
    parser.add_argument(
        "--fps",
        type=float,
        default=5,
        help="server's port number")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="server's port number")

    args = parser.parse_args()
    host_ip = args.host_ip
    port = args.port
    fps = args.fps
    scale = args.scale
    main()
