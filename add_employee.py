import cv2
import face_recognition
import pickle
import os

# التأكد من وجود المجلدات
os.makedirs("employees", exist_ok=True)

# تحميل البيانات السابقة إذا وجدت
encodings_file = "encodings.pickle"
if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# معلومات الموظف
name = input("أدخل اسم الموظف: ")
emp_id = input("أدخل رقم الموظف: ")
department = input("أدخل القسم: ")

# مجلد الموظف
employee_folder = os.path.join("employees", name)
os.makedirs(employee_folder, exist_ok=True)

# فتح الكاميرا
cap = cv2.VideoCapture(0)
num_photos = 5  # عدد الصور الملتقطة تلقائيًا
count = 0
print(f"سيتم التقاط {num_photos} صور للموظف تلقائيًا.")

while count < num_photos:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture Employee", frame)
    key = cv2.waitKey(1)

    if key == ord('s') or count == 0:  
        image_path = os.path.join(employee_folder, f"{emp_id}_{count+1}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"تم حفظ الصورة {count+1} في {image_path}")
        count += 1

    if key == ord('q'):
        print("تم الإلغاء")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ترميز جميع الصور الملتقطة والتحقق من التشابه
for i in range(num_photos):
    image_path = os.path.join(employee_folder, f"{emp_id}_{i+1}.jpg")
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"لم يتم التعرف على الوجه في الصورة {i+1}. حذف الصورة.")
        os.remove(image_path)
        continue

    new_encoding = encodings[0]

    # التحقق من أن الصورة ليست مشابهة جدًا لموظف موجود
    matches = face_recognition.compare_faces(data["encodings"], new_encoding, tolerance=0.45)
    if True in matches:
        print(f"الصورة {i+1} مشابهة لوجه موظف موجود بالفعل! حذف الصورة.")
        os.remove(image_path)
        continue

    data["encodings"].append(new_encoding)
    data["names"].append(name)

# حفظ البيانات المرمزة
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

# حفظ معلومات الموظف في ملف نصي داخل مجلده
info_path = os.path.join(employee_folder, "info.txt")
with open(info_path, "w", encoding="utf-8") as f:
    f.write(f"Name: {name}\nID: {emp_id}\nDepartment: {department}\n")

print("تم إضافة الموظف وترميز الصور بنجاح!")
