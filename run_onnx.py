import cv2
import numpy as np
import argparse
import pandas as pd
import onnxruntime as onnxrt
import cv2
import time
import os


def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    cv2.putText(im, label, (x, y), FONT_FACE, FONT_SCALE, BLACK, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, onnx_session):
      start = time.time()
      input_image = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
      onnx_inputs = {onnx_session.get_inputs()[0].name:input_image.astype(np.float32)}
      onnx_output = onnx_session.run(None, onnx_inputs)
      t = time.time() - start

      return onnx_output, t


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]             
        # Draw bounding box.             
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLACK, 3*THICKNESS)
        # Class label.                      
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])             
        # Draw label.             
        draw_label(input_image, label, left, top)

        list_left.append(left)
        list_top.append(top)
        list_width.append(width)
        list_height.append(height)
        list_conf.append(confidences[i])
        list_class.append(classes[class_ids[i]])

    df_det_post = pd.DataFrame({'left': list_left, 'top': list_top, 'width': list_width, 'height': list_height, 'conf': list_conf, 'classes': list_class})
    print(df_det_post)

    # сохраним детекции в текстовый файл
    if save_txt:
        try:
            os.remove(f"{folder_to_save}/{name.split('.')[0]}.txt")
        except:
            pass
        df_det_post.to_csv(f"{folder_to_save}/{name.split('.')[0]}.txt", header=None, index=None, sep=' ', mode='a')
        folder = folder_to_save.replace('/', '\\')
        print(f"Детекции сохранены здесь: {folder}\{name.split('.')[0]}.txt")

    return input_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='*.png', help='image path')
    parser.add_argument('--weights', type=str, default='*.onnx', help='onnx file')
    parser.add_argument('--classes_file', type=str, default='names.names', help='classes file')
    parser.add_argument('--score_thr', type=float, default=0.5, help='detection threshold')
    parser.add_argument('--nms_thr', type=float, default=0.45, help='overlap area threshold for drop more detections on one place')
    parser.add_argument('--conf_thr', type=float, default=0.45, help='classification threshold')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image sizes')
    parser.add_argument('--folder_to_save', type=str, default='*', help='save to folder/name')
    parser.add_argument('--name', type=str, default='*.png', help='save to folder/name')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()

    # путь к изображению
    img_path = opt.img_path
    # путь к модели onnx
    modelWeights = opt.weights
    # путь загрузки файла имен классов
    classesFile = opt.classes_file
    # ширина входных изображений        
    INPUT_WIDTH = opt.img_size[0]
    # высота входных изображений
    if len(opt.img_size) != 1:
        INPUT_HEIGHT = opt.img_size[1]
    else:
        INPUT_HEIGHT = INPUT_WIDTH
    # порог уверенности классификатора
    SCORE_THRESHOLD = opt.score_thr
    # порог площади перекрытия для исключения лишних рамок
    NMS_THRESHOLD = opt.nms_thr
    # порог уверенности детектора
    CONFIDENCE_THRESHOLD = opt.conf_thr

    # папка для сохранения изображения с детекциями
    folder_to_save = opt.folder_to_save
    # имя файла изображения с детекциями
    name = opt.name
    # сохранять ли текстовый файл с результатами детектирования
    save_txt = opt.save_txt

    # Text parameters
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    THICKNESS = 1
    # Colors
    BLACK  = (0,0,0)
    BLUE   = (255,178,50)
    YELLOW = (0,255,255)

    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
      
    frame = cv2.imread(img_path)
    onnx_session= onnxrt.InferenceSession("*.onnx")

    detections, t = pre_process(frame, onnx_session)
    
    list_left = []
    list_top = []
    list_width = []
    list_height = []
    list_conf = []
    list_class = []
    img = post_process(frame.copy(), detections)
      
    # вычисление времени инференса
    label = 'Inference time: %.2f ms' % (t * 1000.0)
    print(label)

    # сохраним изображение с результатами детектирования
    cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
    cv2.imwrite(f'{folder_to_save}/{name}', img)
    folder = folder_to_save.replace('/', '\\')
    print(f"Результат выполнения кода на изображении можно посмотреть здесь: {folder}\{name}")