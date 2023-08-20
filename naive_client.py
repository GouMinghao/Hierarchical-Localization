from hloc.zmq_handler import ZMQClient, DEFAULT_ADDRESS, encode_img, decode_res
from hloc.visualization import vis_img_with_kpts
import cv2

if __name__ == "__main__":
    zmq_client = ZMQClient(
        address=DEFAULT_ADDRESS,
        encode_fun=encode_img,
        decode_fun=decode_res
    )
    img = cv2.imread("datasets/sacre_coeur/mapping/02928139_3448003521.jpg")
    kpts, scores, descs = zmq_client.request(img)
    print(kpts.shape, scores.shape, descs.shape)
    new_img = vis_img_with_kpts(img, kpts)
    cv2.imshow("sift", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()