from hloc.zmq_handler import ZMQServer, compute_sift, decode_img, encode_res, DEFAULT_ADDRESS
import cv2

def sift_img_callback(message):
    img = decode_img(message)
    compute_sift(img)
    kpts, scores, descs = compute_sift(img)
    message = encode_res(kpts, descs, scores)
    return message

if __name__ == "__main__":
    zmq_server = ZMQServer(
        address=DEFAULT_ADDRESS,
        callback=sift_img_callback
    )
    zmq_server.run()