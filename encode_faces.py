import face_recognition
import cv2
import os
import pickle

# مسار مجلد الموظفين
dataset_path = "employees"
encoding_file = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] جاري معالجة الصور...")

# المرور على مجلد الموظفين
for employee_name in os.listdir(dataset_path):
    emp_dir = os.path.join(dataset_path, employee_name)

    if not os.path.isdir(emp_dir):
        continue

    for img_name in os.listdir(emp_dir):
        img_path = os.path.join(emp_dir, img_name)
        print(f"[INFO] معالجة الصورة: {img_path}")

        # تحميل الصورة
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # تحديد الوجه واستخراج الـ encoding
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(employee_name)

# حفظ النتائج
data = {"encodings": known_encodings, "names": known_names}
with open(encoding_file, "wb") as f:
    pickle.dump(data, f)

print("[INFO] تم حفظ ملف الترميزات بنجاح!")
