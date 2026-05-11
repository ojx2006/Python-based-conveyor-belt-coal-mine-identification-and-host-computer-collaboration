from ultralytics import YOLO
if __name__ == '__main__':
    #加载训练好的权重
    model = YOLO(r'D:\Vision_coal_gangue\runs\coal_v1\weights\best.pt')
    results = model.predict(
        source=r'D:\Vision_coal_gangue\DATASET\train\images\0006_jpeg.rf.5a133a6cb4aca775b9526d74f77a62bd.jpg',
        save=True,
        conf=0.5,
        project='runs',  # 保存的主目录
        name='predict_v1'  # 保存的子目录
    )

    print("预测完成！结果保存在runs/predict_v1 ")