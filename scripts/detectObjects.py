#!/usr/bin/env python
from sys import platform
from cv_bridge import CvBridge, CvBridgeError

import time
import threading
import cv2
import math


def detectWindow(image):
    """"""
    # Detect the red dot above the window
    img = image.copy()
    height, width, _ = img.shape
    red_color_range_ = ((0, 43, 46), (6, 255, 255))

    # 调参点：红点位置的阈值
    x_thereshold = width/2
    y_thereshold = height/2

    # img = cv2.resize(img, (width, height))  # 将图片缩放
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    # cv2.imshow("hsv", cv2.resize(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), (768, 1024)))
    # cv2.waitKey(0)

    img = cv2.inRange(img, red_color_range_[
        0], red_color_range_[1])  # 对原图像和掩模进行位运算
    img = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
    img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

    # 在contours中找出最大轮廓
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # 遍历所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            area_max_contour = c

    isTargetFound = False
    if area_max_contour is not None:
        # 调参点：area_max的容错大小
        if contour_area_max > 10:
            isTargetFound = True

    if not isTargetFound:
        # 没找到目标返回0
        return 0

    if isTargetFound:
        ((centerX, centerY), rad) = cv2.minEnclosingCircle(area_max_contour)
        # 获取了红点的位置
        if(centerX < x_thereshold and centerY < y_thereshold):
            return 1
        elif(centerX < x_thereshold and centerY > y_thereshold):
            return 3
        elif(centerX > x_thereshold and centerY < y_thereshold):
            return 2
        else:
            return 4


# 这个detect_ball(img)函数是接口哦
# img是用cv2.readimage读进来的才行
def load_weight():
    # Initialize this once
    cfg = 'cfg/yolov3.cfg'
    data = 'data/ball.data'
    weights = 'weights/best.pt'
    output = 'data/output'
    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5
    save_txt = False
    save_images = True
    save_path = 'data/output/result.jpg'

    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        s = (320, 192)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    return model, device


def detect_ball(img):
    # Initialized  for every detection
    cfg = 'cfg/yolov3.cfg'
    data = 'data/ball.data'
    weights = 'weights/best.pt'
    output = 'data/output'
    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5
    save_txt = False
    save_images = True
    save_path = 'data/output/result.jpg'
    # Set Dataloader
    img0 = img  # BGR
    model, device = load_weight()
    result = {}

    # Padded resize
    tmpresultimg = letterbox(img0, new_shape=img_size)
    img = tmpresultimg[0]

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]

    # Run inference
    t0 = time.time()

    # Get detections

    img = torch.from_numpy(img).unsqueeze(0).to(device)
    # print("img.shape")
    #print(img.shape )
    pred, _ = model(img)
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]

    if det is not None and len(det) > 0:
        # Rescale boxes from 416 to true image size
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], img0.shape).round()

        # Print results to screen
        # print("image_size")
        # print('%gx%g ' % img.shape[2:])  # print image size
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            print("result")
            print('%g %ss' % (n, classes[int(c)]))
            # return classes[int(c)]

        # Draw bounding boxes and labels of detections
        for det_pack in det:
            # det_pack looks like this:
            # tensor([39.0000,110.0000,169.0000,237.0000,0.99995,0.99995,1.00000])
            xyxy = []
            for index in range(4):
                xyxy.append(float(det_pack[index]))
            # conf是置信度
            conf = det_pack[4]
            # cls是球的类别
            cls = det_pack[6]
            result[classes[int(cls)]] = xyxy[:]
            result[classes[int(cls)]].append(float(conf))
            # #print((xyxy,conf, cls_conf, cls ))
            # if save_txt:  # Write to file
            #     with open(save_path + '.txt', 'a') as file:
            #         file.write(('%g ' * 6 + '\n') % (xyxy, cls, conf))

            # # Add bbox to the image
            # label = '%s %.2f' % (classes[int(cls)], conf)
            # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
            # #cv2.imshow('result',img0)
            #  #cv2.waitKey()
            # if save_images:  # Save image with detections
            #     storepath='result/ball_detect'+'test'+'.jpg'
            #     cv2.imwrite(storepath, img0)

    return result
    #print('Done. (%.3fs)' #


def letterbox(img, new_shape=416, color=(128, 128, 128), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    if shape[::-1] != new_unpad:  # resize
        # INTER_AREA is better, INTER_LINEAR is faster
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return (img, ratiow, ratioh, dw, dh)


# #detect_ball的测试代码主函数
# if __name__=="__main__":
#     path="/home/olivia/catkin_ws/src/image_tran/scripts/yolov3_detect/data/samples/three_balls_tept.jpg"
#     img=cv2.imread(path)
#     result_obj=detect_ball(img)
#     print(result_obj)

# #detect_ball的测试代码主函数
# if __name__=="__main__":
#     path="/home/olivia/catkin_ws/src/image_tran/scripts/yolov3_detect/data/samples/red_dot_test2.jpg"
#     img=cv2.imread(path)
#     print(detectWindow(img))
