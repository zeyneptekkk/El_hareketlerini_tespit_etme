import cv2
import mediapipe as mp

# MediaPipe araçlarını tanımlama
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam'den görüntü alma
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5,
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Bir hata oluştu")
            continue

        # Görüntüyü işle
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Görüntüyü tekrar yazılabilir yap ve BGR'ye dönüştür
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Landmarks'ları çizmee
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Sonucu gösterme
        cv2.imshow("Result", image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC tuşu ile çıkış
            break


cap.release()
cv2.destroyAllWindows()