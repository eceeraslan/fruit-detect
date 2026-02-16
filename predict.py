from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/runs/orange_freshness5/weights/best.pt')
    

    results = model.predict(
        source='dataset/test/images',
        conf=0.5,
        save=True,
        project='runs',
        name='test_predictions'
    )
    
    print("Tahminler kaydedildi: runs/test_predictions/")