import numpy as np
import zmq

DEFAULT_ADDRESS = "tcp://0.0.0.0:5556"

"""
## Image protocal

Header: [int32(height), int32(width), int32(channels)]
Content: [uint8 * height * width * channel (HWC) image in BGR order]

## Response protocal

Header: [int32(num_kpts), int32(desc_dim), int32(desc_type): {0: uint8, 1: float32}]
Content: [
    float32 * 2 * num_kpts(xy order),
    float32 * num_kpts,
    desc_type * num_kpts * desc_dim
]
"""

DESC_TYPE_INVERSE_DICT = {
    0: np.uint8,
    1: np.float32,
}

def encode_img(img):
    """encode img to zmq message

    Args:
        img(np.array): numpy image(BGR)

    Returns: 
        bytes: zmq messages
    """
    assert len(img.shape) in [2, 3]
    assert img.dtype == np.uint8
    if len(img.shape) == 2:
        channels = 1
        height, width = img.shape
    if len(img.shape) == 3:
        channels = 3
        height, width, c = img.shape
        assert c == channels
    header = np.array([height, width, channels], dtype=np.int32).tobytes()
    content = img.tobytes()
    msg = header + content
    return msg

def decode_img(message):
    """decode img from zmq

    Args:
        message (bytes): zmq messages

    Returns:
        np.array: numpy image(BGR)
    """
    header = np.frombuffer(message[:12], dtype=np.int32)
    img = np.copy(np.frombuffer(message[12:], dtype=np.uint8))
    assert header[2] in [1, 3]
    if header[2] == 1:
        # gray
        img = img.reshape((header[0], header[1]))
    elif header[2] == 3:
        # BGR
        img = img.reshape((header[0], header[1], header[2]))
    return img

def encode_res(kpts, descs, scores):
    """encode result to zmq message

    Args:
        kpts (numpy.array[num_kpt, 2]): keypoint coordinates
        descs (numpy.array[num_kpt, desc_dim]): descriptors
        scores (numpy.array[num_kpt]): scores

    Returns:
        bytes: zmq results
    """  
    # Header: [int32(num_kpts), int32(desc_dim), int32(desc_type): {0: uint8, 1: float32}]
    # Content: [
    #     float32 * num_kpts * 2 (xy order),
    #     float32 * num_kpts,
    #     desc_type * num_kpts * desc_dim
    # ]
    if descs.dtype == np.uint8:
        desc_type_int = 0
    elif descs.dtype == np.float32:
        desc_type_int = 1
    else:
        return TypeError("Unknown desc type")
    assert len(scores.shape) == 1
    assert len(descs.shape) == 2
    assert len(kpts.shape) == 2
    num_kpt, desc_dim = descs.shape
    assert scores.shape[0] == num_kpt
    assert kpts.shape[0] == num_kpt
    assert kpts.shape[1] == 2
    header = np.array([num_kpt, desc_dim, desc_type_int], dtype=np.int32).tobytes()
    kpt_coords_bytes = kpts.astype(np.float32).tobytes()
    scores_bytes = scores.astype(np.float32).tobytes()
    descs_bytes = descs.tobytes()
    msg = header + kpt_coords_bytes + scores_bytes + descs_bytes
    return msg

def decode_res(message):
    """decode result from zmq message

    Args:
        message(bytes): zmq results

    Returns:
        kpts (numpy.array): keypoint coordinates
        descs (numpy.array): descriptors
        scores (numpy.array): scores
    """
    header = np.frombuffer(message[:12], dtype=np.int32)
    num_kpts, desc_dim = header[:2]
    desc_type = DESC_TYPE_INVERSE_DICT[header[2]]
    kpts_len = num_kpts * 2 * 4 # 2 for xy and 4 for float32
    scores_len = num_kpts * 4 # 4 for float32
    kpts = np.copy(np.frombuffer(message[12: 12 + kpts_len], dtype=np.float32)).reshape((num_kpts, 2))
    scores = np.copy(np.frombuffer(message[12 + kpts_len: 12 + kpts_len + scores_len],
        dtype=np.float32))
    descs = np.copy(np.frombuffer(message[12 + kpts_len + scores_len:], dtype=desc_type)).reshape((num_kpts, desc_dim))
    return kpts, descs, scores,
        

class ZMQServer():
    def __init__(self, address, callback):
        self.address = address
        self.callback = callback
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(DEFAULT_ADDRESS)

    def run(self):
        while True:
            message = self.socket.recv()
            back_message = self.callback(message)
            self.socket.send(back_message)

class ZMQClient():
    def __init__(self, address, encode_fun, decode_fun):
        self.address = address
        self.encode_fun = encode_fun
        self.decode_fun = decode_fun
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(DEFAULT_ADDRESS)
    
    def request(self, data):
        self.socket.send(self.encode_fun(data))
        return self.decode_fun(self.socket.recv())

def compute_sift(img):
    import cv2
    sift = cv2.SIFT_create()
    kpts, descs = sift.detectAndCompute(img, None)
    assert len(kpts) == descs.shape[0]
    kpt_list = []
    score_list = []
    for i in range(len(kpts)):
        kpt_list.append(kpts[i].pt)
        score_list.append(kpts[i].response)
    return np.array(kpt_list, dtype=np.float32), np.array(score_list, dtype=np.float32), descs

if __name__ == "__main__":
    import cv2
    IMG_PATH = "datasets/sacre_coeur/mapping/02928139_3448003521.jpg"
    # test gray image
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    message = encode_img(img)
    new_img = decode_img(message)
    np.testing.assert_array_equal(img, new_img)

    # test BGR image
    img = cv2.imread(IMG_PATH)
    message = encode_img(img)
    new_img = decode_img(message)
    np.testing.assert_array_equal(img, new_img)

    # test gray image
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    kpts, scores, descs = compute_sift(img)
    print("kpts.shape: {}, scores.shape: {}, descs shape: {}".format(
        kpts.shape, scores.shape, descs.shape
    ))
    print("kpts.dtype: {}, scores.dtype: {}, descs dtype: {}".format(
        kpts.dtype, scores.dtype, descs.dtype
    ))
    message = encode_res(kpts, descs, scores)
    new_kpts, new_descs, new_scores = decode_res(message)
    np.testing.assert_array_equal(new_kpts, kpts)
    np.testing.assert_array_equal(new_descs, descs)
    np.testing.assert_array_equal(new_scores, scores)