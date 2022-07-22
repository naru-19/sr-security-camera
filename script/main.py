# streamlit run main.py

import streamlit as st
from numpysocket import NumpySocket
import cv2
import time
import numpy as np
from PIL import Image

def main():
    print("start")
    start = time.time()
    st.markdown("# Camera")
    np_sock = NumpySocket()
    np_sock.startServer(41234)
    time_loc = st.empty()
    image_loc = st.empty()
    
    while True:
        image_raw = np_sock.recieve()
        time_loc.text(str(time.time() - start))
        image_rgb = Image.fromarray(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
        image_loc.image(image_rgb)

        if cv2.waitKey() & 0xFF == ord("q"):
            break

    np_sock.close()

if __name__ == "__main__":
    main()


# import streamlit as st
# from npsocket import SocketNumpyArray

# sock_receiver = SocketNumpyArray()

# def main():
#     st.markdown("# Camera")
#     image_loc = st.empty()
#     if 'connected' not in st.session_state:
#         st.session_state['connected'] = False

#     def init_sock_receiver():
#         st.session_state['connected'] = False
#         if st.session_state.cam_port.isnumeric():
#             sock_receiver.initalize_receiver(int(st.session_state.cam_port))
#             st.session_state['connected'] = True
#         else:
#             st.write("Write port number.")

#     st.text_input("input port for recive camera stream", "41234", key="cam_port")
#     st.button(label='Connection', on_click=init_sock_receiver)

#     while st.session_state['connected']:
#         image_raw = sock_receiver.receive_array()
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         image_loc.image(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break


# if __name__ == "__main__":
#     main()