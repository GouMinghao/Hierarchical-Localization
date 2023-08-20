import zmq
from . import extract_features, logger
from .zmq_handler import ZMQServer, DEFAULT_ADDRESS
import torch
import numpy as np
import cv2

DEFAULT_CONF = extract_features.confs['superpoint_inloc']

class HlocExtractorZMQServer():
    def __init__(self, conf, address = DEFAULT_ADDRESS, device="cuda:0"):
        self.address = address
        self.zmq_handler = ZMQServer(address = self.address)
        self.device = device
        self.conf = conf
        self.model_type = extract_features.dynamic_load(extract_features.extractors, self.conf["model"]["name"])
        self.model = self.model_type(conf['model']).eval().to(self.device)

    def infer_one(self, img):
        # convert numpy img to torch format
        torch_img = img


        pred = self.model({'image': torch_img.to(self.device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        # pred['image_size'] = original_size = data['original_size'][0].numpy()
        
        # if 'keypoints' in pred:
        #     size = np.array(torch_img.shape[-2:][::-1])
        #     scales = (original_size / size).astype(np.float32)
        #     pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
        #     if 'scales' in pred:
        #         pred['scales'] *= scales.mean()
        #     # add keypoint uncertainties scaled to the original resolution
        #     uncertainty = getattr(self.model, 'detection_noise', 1) * scales.mean()

        # if as_half:
        #     for k in pred:
        #         dt = pred[k].dtype
        #         if (dt == np.float32) and (dt != np.float16):
        #             pred[k] = pred[k].astype(np.float16)

        # with h5py.File(str(feature_path), 'a', libver='latest') as fd:
        #     try:
        #         if name in fd:
        #             del fd[name]
        #         grp = fd.create_group(name)
        #         for k, v in pred.items():
        #             grp.create_dataset(k, data=v)
        #         if 'keypoints' in pred:
        #             grp['keypoints'].attrs['uncertainty'] = uncertainty
        #     except OSError as error:
        #         if 'No space left on device' in error.args[0]:
        #             logger.error(
        #                 'Out of disk space: storing features on disk can take '
        #                 'significant space, did you enable the as_half flag?')
        #             del grp, fd[name]
        #         raise error
        logger.info('Finished exporting features.')
        return pred["keypoints"], pred["descriptors"], pred["scores"]

    def run(self):
        """main loop of server
        """
        while True:
            # recv from client
            message = self.socket.recv()

            # convert message to numpy image
            img = self._decode_img(message)

            # get result
            kpts, descs, scores = self.infer_one(img)

            # encode message
            b_res = self._encode_res(kpts, descs, scores)
            
            # send response to client
            self.socket.send(b_res)
