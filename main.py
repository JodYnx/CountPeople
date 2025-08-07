import cv2
from ultralytics import YOLO
import os


class AccuratePeopleCounter:
    def __init__(self):
        # تحميل النموذج المتوسط (yolov8m.pt)
        self.model = YOLO('yolov8m.pt')
        self.unique_persons = set()  # لتخزين الهويات الفريدة
        self.frame_count = 0
        self.total_persons = 0  # العدد الإجمالي دون تكرار

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"خطأ: لا يمكن فتح الفيديو {video_path}")
            return

        print("جاري تحليل الفيديو... (سيتوقف تلقائيًا عند النهاية)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:  # توقف عند نهاية الفيديو
                break

            self.frame_count += 1

            # استخدام النموذج مع إعدادات محسنة
            results = self.model.track(
                frame,
                persist=True,
                classes=[0],  # أشخاص فقط
                conf=0.7,  # عتبة ثقة أعلى
                iou=0.45,  # تقليل التداخل
                tracker="bytetrack.yaml",
                verbose=False
            )

            # تحديث العد الفريد
            if results[0].boxes.id is not None:
                current_ids = set(map(int, results[0].boxes.id.cpu().numpy()))
                new_persons = current_ids - self.unique_persons
                self.total_persons += len(new_persons)
                self.unique_persons.update(current_ids)

            # عرض النتائج (اختياري)
            self.display_info(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # إيقاف يدوي
                break

        cap.release()
        cv2.destroyAllWindows()
        self.show_final_results()

    def display_info(self, frame):
        """عرض المعلومات على الإطار"""
        cv2.putText(frame, f"Detected Persons: {len(self.unique_persons)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Count:{self.total_persons}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Output", frame)

    def show_final_results(self):
        """عرض النتائج النهائية"""
        print("\n" + "=" * 50)
        print("النتائج النهائية الدقيقة:")
        print(f"عدد الإطارات المعالجة: {self.frame_count}")
        print(f"عدد الأشخاص المختلفين: {len(self.unique_persons)}")
        print(f"العدد التراكمي: {self.total_persons}")
        print("=" * 50)


if __name__ == "__main__":
    counter = AccuratePeopleCounter()
    video_path = "data/input/3202044-hd_1920_1080_25fps.mp4"

    # إنشاء المجلدات اللازمة
    os.makedirs("data/input", exist_ok=True)

    counter.process_video(video_path)