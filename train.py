from  ultralytics  import YOLO



if __name__=='__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data = 'dataset/data.yaml',
        workers=4,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project='runs',
        name='orange_freshness'
    )