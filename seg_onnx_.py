import onnxruntime
import numpy as np
import cv2


# 调色板
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

# 4个关键点连接顺序
skeleton = [[1, 2], [2, 3], [3, 4], [4, 1]]

# 骨架颜色
pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# 关键点颜色
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def LetterBox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    '''  调整图像大小和两边灰条填充  '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    if not scaleup:
        r = min(r, 1.0)
    # 计算pad长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2
    dh /= 2
    # 将原图resize到new_unpad（长边相同，比例相同的新图）
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下两侧的padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算左右两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加灰条
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def PreProcess(img):
    # 归一化 调整通道为（1，3，640，640）
    img = img / 255.
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    return data


def xywh2xyxy(x):
    ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# nms算法
def NMS(dets, iou_thresh):
    # dets: N * M, N是bbox的个数，M的前4位是对应的 左上点，右下点
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= iou_thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return np.array(output)


def xyxy2xywh(a):
    ''' 左上点 右下点 ------>>> 左上点 宽 高 '''
    b = np.copy(a)
    # y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    b[:, 2] = a[:, 2] - a[:, 0]  # w
    b[:, 3] = a[:, 3] - a[:, 1]  # h
    return b


def ScaleBoxes(img1_shape, boxes, img0_shape):
    '''   将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param boxes:  预测的box信息
    :param img0_shape: 原始图像尺度
    '''
    # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    num_kpts = boxes.shape[1] // 3  # 56 // 3 = 18
    for kid in range(2, num_kpts + 1):
        boxes[:, kid * 3 - 1] = (boxes[:, kid * 3 - 1] - pad[0]) / gain
        boxes[:, kid * 3] = (boxes[:, kid * 3] - pad[1]) / gain
    # boxes[:, 5:] /= gain  # 关键点坐标还原到原图上
    ClipBoxes(boxes, img0_shape)
    return boxes


def ClipBoxes(boxes, shape):
    # 进行一个边界截断，以免溢出
    # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x  # 左上
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x  # 右下
    boxes[:, 3] = bottom_right_y


def PlotSkeletonKpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps  # 51 / 3 =17
    # 画点
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:  # 关键点的置信度必须大于 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), 10, (int(r), int(g), int(b)), -1)
    # 画骨架
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if conf1 > 0.5 and conf2 > 0.5:  # 相连的两个关键点置信度 必须同时大于 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


class KeyPoint():
    def __init__(self, modelpath, imgsz=None, show=None, threshold=0.5):
        # self.session = onnxruntime.InferenceSession(modelpath, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.session = onnxruntime.InferenceSession(modelpath, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        self.conf = threshold
        if imgsz==None:
            self.imgsz = (640, 640)
        else:
            self.imgsz = imgsz
        if show == None:
            self.show = False
        else:
            self.show = show

    def inference(self, image):
        '''
            input: cv2 image
            output: [{'bbox':[x1, y1, x2, y2],
                    'conf': bbox_confidence,
                    'keypoints': [x1, y1, keypoint1_conf, x2, y2, keypoint2_conf, ...]}]
        '''
        img = LetterBox(image, new_shape=self.imgsz)
        data = PreProcess(img)
        # 预测输出float32[1, 116, 8400]
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        # [116, 8400]
        pred = pred[0]
        # [8400, 116]
        pred = np.transpose(pred, (1, 0))
        # 置信度阈值过滤
        pred = pred[pred[:, 4] > self.conf]
        if len(pred) == 0:
            print("没有检测到任何关键点")
            return
        else:
            # 中心宽高转左上点，右下点
            bboxs = xywh2xyxy(pred)
            # NMS处理
            bboxs = NMS(bboxs, iou_thresh=0.6)
            # 坐标从左上点，右下点 到 左上点，宽，高.
            bboxs = np.array(bboxs)
            bboxs = xyxy2xywh(bboxs)
            # 坐标点还原到原图
            bboxs = ScaleBoxes(img.shape, bboxs, image.shape)

            results = []
            for box in bboxs:
                # 依次为 检测框（左上点，右下点）、置信度、4个关键点
                det_bbox, det_scores, kpts = box[0:4], box[4], box[5:]
                result={'bbox': det_bbox.tolist(), 'conf': det_scores, 'keypoints': kpts.tolist()}
                results.append(result)
                if self.show:

                    # 画框
                    cv2.rectangle(image, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])),
                                  (0, 0, 255), 2)
                    # 边界框检测置信度
                    if int(det_bbox[1]) < 30:
                        cv2.putText(image, "conf:{:.2f}".format(det_scores), (int(det_bbox[0]) + 5, int(det_bbox[1]) + 25),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
                    else:
                        cv2.putText(image, "conf:{:.2f}".format(det_scores), (int(det_bbox[0]) + 5, int(det_bbox[1]) - 5),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
                    # 画点 连线
                    PlotSkeletonKpts(image, kpts)
            if self.show:
                cv2.namedWindow("keypoint", cv2.WINDOW_NORMAL)
                cv2.imshow("keypoint", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return results


if __name__ == '__main__':
    # 1.实例化模型
    model_path = r'./yolov8s-seg.onnx'
    keydet = KeyPoint(model_path, show=True, imgsz=(640, 640), threshold=0.65)

    # 2.读取图片
    image = cv2.imread('./test.jpg')

    # 3.输出关键点检测推理结果
    # 格式：
    # [{'bbox': [x1, y1, x2, y2],
    #   'conf': bbox_confidence,
    #   'keypoints': [x1, y1, keypoint1_conf, x2, y2, keypoint2_conf, ...]},]
    result = keydet.inference(image)


