import torch

def main():

    img_path = str(input('Введите путь к изображению: '))
    if img_path == '':
        img_path = '*.png'
        print(img_path)

    model_path = str(input('Введите путь к модели детектора yolov5 *.pt: '))
    if model_path == '':
        model_path = '*.pt'
        print(model_path)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, force_reload=True)

    results = model(img_path)
    print(results.pandas().xyxy[0])

if __name__ == '__main__':
    main()
