import cv2
import numpy as np
import os
import glob
import torch
from torchvision.transforms.functional import normalize
from facexlib.detection import init_detection_model
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor, imwrite
from face_detection.utils.face_util import *
from narutils import *


class FaceDetector(object):
    """Face detection pipeline"""

    def __init__(self,
                 upscale_factor,
                 face_size=128,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None):
        """
        :param upscale_factor: GANのアップスケール倍率
        :param face_size: GANの入力にする画像のサイズ
        :param crop_ratio: 変えなくていい
        :param det_model: retinaface_resnet50 or retinaface_mobile0.25
        :param save_ext: pngとか
        :param template_3points: いらん
        :param pad_blur: いらん
        :param use_parse: いらん
        :param device: デフォルトはcuda:0
        """
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = upscale_factor
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))

        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])
        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.input_img = None
        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.bboxes = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # init face detection model
        self.face_det = init_detection_model(det_model, half=False, device=self.device)

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = init_parsing_model(model_name='parsenet', device=self.device)

    def crop_faces(self, input_img, save_path=None):
        """
        :param input_img: [np.array | str] 顔が映った画像1枚
        :param save_path: [str] 保存先　なくてもいい
        :return: cropped_faces: [list of np.array] クロップされた顔（複数）
        :return: bboxes: [list of np.array] クロップに対応した4点（複数）
        """
        self.clean_all()
        self.read_image(input_img)
        self.get_face_landmarks_5()
        self.align_warp_face(save_path)
        self.get_face_bboxes()
        return self.cropped_faces, self.bboxes

    def restore_faces_in_input_image(self, restored_faces, save_path=None):
        """
        :param restored_faces: [list of np.array] 超解像された顔画像
        :param save_path: [str] 保存先　なくてもいい
        :return: upsample_img [np.array] 顔が超解像された全体画像
        """
        w_up, h_up = int(self.face_size[0] * self.upscale_factor), int(self.face_size[1] * self.upscale_factor)
        for restored_face in restored_faces:
            self.add_restored_face(cv2.resize(restored_face, (w_up, h_up), interpolation=cv2.INTER_LINEAR))
        upsample_img = self.paste_faces_to_input_image(save_path)
        return upsample_img

    def read_image(self, img):
        """img can be image path or cv2 loaded image."""
        # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
        if isinstance(img, str):
            img = cv2.imread(img)

        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img = img[:, :, 0:3]

        self.input_img = img

    def get_face_landmarks_5(self,
                             only_keep_largest=False,
                             only_center_face=False,
                             resize=None,
                             blur_ratio=0.01,
                             eye_dist_threshold=None):
        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = min(h, w) / resize
            h, w = int(h / scale), int(w / scale)
            input_img = cv2.resize(self.input_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        with torch.no_grad():
            detect_bboxes = self.face_det.detect_faces(input_img, 0.97) * scale
        for bbox in detect_bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue

            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]

        # pad blurry images
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # get landmarks
                eye_left = landmarks[0, :]
                eye_right = landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg

                # Get the oriented crop rectangle
                # x: half width of the oriented crop rectangle
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
                # norm with the hypotenuse: get the direction
                x /= np.hypot(*x)  # get the hypotenuse of a right triangle
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                # y: half height of the oriented crop rectangle
                y = np.flipud(x) * [-1, 1]

                # c: center
                c = eye_avg + eye_to_mouth * 0.1
                # quad: (left_top, left_bottom, right_bottom, right_top)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                # qsize: side length of the square
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)

                # get pad
                # pad: (width_left, height_top, width_right, height_bottom)
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                       int(np.ceil(max(quad[:, 1]))))
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[0] + border, 1),
                    max(pad[3] - self.input_img.shape[1] + border, 1)
                ]

                if max(pad) > 1:
                    # pad image
                    pad_img = np.pad(self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    # modify landmark coords
                    landmarks[:, 0] += pad[0]
                    landmarks[:, 1] += pad[1]
                    # blur pad images
                    h, w, _ = pad_img.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                       np.float32(w - 1 - x) / pad[2]),
                                      1.0 - np.minimum(np.float32(y) / pad[1],
                                                       np.float32(h - 1 - y) / pad[3]))
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0:
                        blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    # blur_img = cv2.GaussianBlur(pad_img, (blur, blur), 0)

                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) - pad_img) * np.clip(mask, 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255)  # float32, [0, 255]
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))

        return len(self.all_landmarks_5)

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp faces with face template.
        """
        if self.pad_blur:
            assert len(self.pad_input_imgs) == len(
                self.all_landmarks_5), f'Mismatched samples: {len(self.pad_input_imgs)} and {len(self.all_landmarks_5)}'
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            # use cv2.LMEDS method for the equivalence to skimage transform
            # ref: https://blog.csdn.net/yichxi/article/details/115827338
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            cropped_face = cv2.warpAffine(
                input_img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                save_path = f'{os.path.splitext(save_cropped_path)[0]}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Get inverse affine matrix."""
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                save_path = f'{os.path.splitext(save_inverse_affine_path)[0]}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def get_face_bboxes(self):
        self.get_inverse_affine()
        for cropped_face, inverse_affine_matrix in zip(self.cropped_faces, self.inverse_affine_matrices):
            height, width = cropped_face.shape[:2]
            # 右周りの点(opencvのcontourに合わせている)
            cropped_rect = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            cropped_rect = np.array([cropped_rect], dtype=float)
            inv_affine = np.vstack([inverse_affine_matrix, np.array([0, 0, 1])])
            # opencvのcontourと同じshapeにしている
            self.bboxes.append(cv2.perspectiveTransform(cropped_rect, inv_affine).reshape(-1, 1, 2).astype(int))

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, save_path=None, upsample_img=None):
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        if upsample_img is None:
            # simply resize the background
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), 'length of restored_faces and affine_matrices are different.'
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # Add an offset to inverse affine matrix, for more precise back alignment
            if self.upscale_factor > 1:
                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

            if self.use_parse:
                # inference
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)
                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()

                mask = np.zeros(out.shape)
                mask_colormap = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for idx, color in enumerate(mask_colormap):
                    mask[out == idx] = color
                #  blur the mask
                mask = cv2.GaussianBlur(mask, (101, 101), 11)
                mask = cv2.GaussianBlur(mask, (101, 101), 11)
                # remove the black borders
                threshold = 10
                mask[:threshold, :] = 0
                mask[-threshold:, :] = 0
                mask[:, :threshold] = 0
                mask[:, -threshold:] = 0
                mask = mask / 255.

                mask = cv2.resize(mask, restored_face.shape[:2])
                mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)
                inv_soft_mask = mask[:, :, None]
                pasted_face = inv_restored

            else:  # use square parse maps
                mask = np.ones(self.face_size, dtype=np.float32)
                inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
                # remove the black borders
                inv_mask_erosion = cv2.erode(
                    inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
                pasted_face = inv_mask_erosion[:, :, None] * inv_restored
                total_face_area = np.sum(inv_mask_erosion)
                # compute the fusion edge based on the area of face
                w_edge = int(total_face_area ** 0.5) // 20
                erosion_radius = w_edge * 2
                inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
                blur_size = w_edge * 2
                inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
                if len(upsample_img.shape) == 2:  # upsample_img is gray image
                    upsample_img = upsample_img[:, :, None]
                inv_soft_mask = inv_soft_mask[:, :, None]

            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        if save_path is not None:
            save_path_tmp = f'{os.path.splitext(save_path)[0]}.{self.save_ext}'
            imwrite(upsample_img, save_path_tmp)
        return upsample_img

    def clean_all(self):
        self.input_img = None
        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.bboxes = []
        self.restored_faces = []
        self.pad_input_imgs = []


def test():
    # initialize face helper
    face_detector = FaceDetector(upscale_factor=4, face_size=128)

    img_paths = glob.glob('./test/*.jpg')  # 適当な顔画像を置く
    save_path = 'test/output'
    for i, path in enumerate(img_paths):
        img = cv2.imread(path)
        print(i, path)
        file_name = os.path.basename(path)
        save_path = opj(save_path, file_name)
        cropped_faces, bboxes = face_detector.crop_faces(img, save_path=save_path)
        img_drawed = img.copy()
        img_drawed = cv2.drawContours(img_drawed, bboxes, -1, (0, 255, 0), thickness=2)
        cv2.imwrite(f"{os.path.splitext(save_path)[0]}_{i:02d}_bbox.png", img_drawed)
        face_detector.restore_faces_in_input_image(cropped_faces, save_path=save_path)


if __name__ == '__main__':
    test()
