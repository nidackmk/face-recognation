import cv2
import time
import csv

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()  # Frame'in kopyasını oluştur
    frameHeight = frameOpencvDnn.shape[0]  # Frame yüksekliğini al
    frameWidth = frameOpencvDnn.shape[1]  # Frame genişliğini al
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)  # Görüntüyü blob formatına dönüştür

    net.setInput(blob)  # Blob'u ağın girişine ver
    detections = net.forward()  # Yüz tespitlerini yap
    bboxes = []  # Tespit edilen yüzlerin kutularını saklayacak liste

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Tespit edilen yüzün güven skorunu al
        if confidence > conf_threshold:  # Güven skoru eşik değerinden büyükse
            x1 = int(detections[0, 0, i, 3] * frameWidth)  # Yüz kutusunun sol üst x koordinatını hesapla
            y1 = int(detections[0, 0, i, 4] * frameHeight)  # Yüz kutusunun sol üst y koordinatını hesapla
            x2 = int(detections[0, 0, i, 5] * frameWidth)  # Yüz kutusunun sağ alt x koordinatını hesapla
            y2 = int(detections[0, 0, i, 6] * frameHeight)  # Yüz kutusunun sağ alt y koordinatını hesapla
            bboxes.append([x1, y1, x2, y2])  # Yüz kutusunu listeye ekle
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)  # Yüz kutusunu çiz

    return frameOpencvDnn, bboxes  # Tespit edilen yüz kutularını ve işlenmiş frame'i döndür

# Model ve konfigürasyon dosyaları
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Ortalama değerler ve yaş/cinsiyet listeleri
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Erkek', 'Kadin']

# Yaş kategorileri
ageCategories = {
    '(0-2)': 'Bebek',
    '(4-6)': 'Cocuk',
    '(8-12)': 'Cocuk',
    '(15-20)': 'Genc',
    '(25-32)': 'Genc',
    '(38-43)': 'Yetiskin',
    '(48-53)': 'Yetiskin',
    '(60-100)': 'Yasli'
}

# Ağları yükle
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Video yolunu belirliyoruz
video_path = "videos/son.mp4"
cap = cv2.VideoCapture(video_path)  # Videoyu aç
padding = 20  # Yüz kutusunun etrafındaki boşluk

# VideoWriter nesnesini oluştur (MP4 formatında)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# CSV dosyasını oluştur ve başlıkları yaz
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Gender", "Age", "Age Category", "Confidence"])  # Başlıkları yaz

    while True:
        t = time.time()  # İşlem süresini ölçmek için başlangıç zamanı
        hasFrame, frame = cap.read()  # Videodan bir frame oku
        if not hasFrame:
            break  # Frame yoksa döngüden çık

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Frame'i yarı boyutuna küçült
        frameFace, bboxes = getFaceBox(faceNet, small_frame)  # Yüz tespitlerini yap

        if not bboxes:
            print("Yüz bulunamadı")  # Yüz tespit edilemezse mesaj yaz
            continue  # Sonraki frame'e geç

        for bbox in bboxes:
            face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]  # Yüz bölgesini al
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  # Yüzü blob formatına dönüştür

            # Cinsiyet tahmini
            genderNet.setInput(blob)  # Blob'u ağın girişine ver
            genderPreds = genderNet.forward()  # Cinsiyet tahminlerini al
            gender = genderList[genderPreds[0].argmax()]  # En yüksek skora sahip cinsiyeti seç
            gender_conf = genderPreds[0].max()  # Cinsiyet güven skorunu al
            print("Gender : {}, conf = {:.3f}".format(gender, gender_conf))  # Cinsiyet ve güven skorunu yazdır

            # Yaş tahmini
            ageNet.setInput(blob)  # Blob'u ağın girişine ver
            agePreds = ageNet.forward()  # Yaş tahminlerini al
            age = ageList[agePreds[0].argmax()]  # En yüksek skora sahip yaşı seç
            age_conf = agePreds[0].max()  # Yaş güven skorunu al
            print("Age Output : {}".format(agePreds))  # Yaş tahmin çıktısını yazdır
            print("Age : {}, conf = {:.3f}".format(age, age_conf))  # Yaş ve güven skorunu yazdır

            # Yaş kategorisini belirle
            ageCategory = ageCategories[age]  # Yaş kategorisini belirle
            label = "{}, {}, {}".format(gender, age, ageCategory)  # Etiketi oluştur
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # Etiketi yüz kutusunun üstüne yaz

            # CSV dosyasına yaz
            writer.writerow([gender, age, ageCategory, max(gender_conf, age_conf)])  # Tahmin sonuçlarını CSV'ye yaz
        
        # Yazılan frame'i VideoWriter nesnesine ekle
        out.write(frameFace)
        cv2.imshow("Age Gender Demo", frameFace)  # İşlenmiş frame'i göster

        print("time : {:.3f}".format(time.time() - t))  # İşlem süresini yazdır
        
        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()  # Video dosyasını kapat
out.release()  # VideoWriter nesnesini serbest bırak
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat
