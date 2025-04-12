#IMPORT LIBRARY
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

#FUNGSI UNTUK MEMBACA DATA GAMBAR
def load_image(image_path):
  image = cv2.imread(image_path)
  if image is None:
    print('Error: Tidak bisa membaca gambar')
    return None, None
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return image, gray

#MEMBACA DAN MEMBUAT SAMPEL GAMBAR
sample_image, sample_image_gray = load_image('images/George_W_Bush/1.jpg')
sample_image_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

#DEFINISI NAMA FOLDER, GAMBAR DAN LABEL
dataset_dir = 'images'
images = []
labels = []

#LOOPING UNTUK MEMBACA GAMBAR DAN MENAMBAHKAN LABELS BERDASARKAN FOLDER
for root, dirs, files in os.walk(dataset_dir):
  if len(files) == 0:
    continue
  for f in files:
    _, image_gray = load_image(os.path.join(root, f))
    if image_gray is None:
      continue
    images.append(image_gray)
    labels.append(os.path.basename(root))

#INISIASI HAAR CASCADE DAN FUNGSI UNTUK MENDETEKSI WAJAH DENGAN HAAR CASCADE
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
  faces = face_cascade.detectMultiScale(
    image_gray,
    scaleFactor=scale_factor,
    minNeighbors=min_neighbors,
    minSize=min_size
  )
  return faces


#FUNGSI UNTUK CROP GAMBAR WAJAH TERDETEKSI
def crop_faces(image_gray, faces, return_all=False):
  cropped_faces = []
  selected_faces = []
  if len(faces) > 0:
    if return_all:
      for x, y, w, h in faces:
        selected_faces.append((x, y, w, h))
        cropped_faces.append(image_gray[y:y+h, x:x+w])
    else:
      x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
      selected_faces.append((x, y, w, h))
      cropped_faces.append(image_gray[y:y+h, x:x+w])
  return cropped_faces, selected_faces


#MENGUBAH UKURAN GAMBAR WAJAH MENJADI 128x128 DAN MELAKUKAN FLATTENING MENJADI VEKTOR 1D
face_size = (128, 128)
def resize_and_flatten(face):
  face_resized = cv2.resize(face, face_size)
  face_flattened = face_resized.flatten()
  return face_flattened

#DEFINISI VARIABEL UNTUK MENYIMPAN GAMBAR WAJAH DAN LABEL
X = []
y = []

#LOOPING UNTUK MENDETEKSI WAJAH DAN MENYIMPAN GAMBAR WAJAH YANG TELAH DICROP
for image, label in zip(images, labels):
  faces = detect_faces(image)
  cropped_faces, _ = crop_faces(image, faces)
  if len(cropped_faces) > 0:
    face_flattened = resize_and_flatten(cropped_faces[0])
    X.append(face_flattened)
    y.append(label)

#MENGUBAH LIST GAMBAR WAJAH DAN LABEL MENJADI ARRAY NUMPY
X = np.array(X)
y = np.array(y)

#MEMBAGI DATA MENJADI DATA LATIH DAN DATA UJI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=177, stratify=y)

#MEAN CENTERING
class MeanCentering(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    self.mean_face = np.mean(X, axis=0)
    return self

  def transform(self, X):
    return X - self.mean_face

#MEMBUAT CLASSIFIER PIPELINE
pipe = Pipeline([
     ('centering', MeanCentering()),
     ('pca', PCA(svd_solver='randomized', whiten=True,random_state=177)),
     ('svc', SVC(kernel='linear', random_state=177))
])

#LATIH DAN MENAMPILKAN EVALUASI MODEL
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

#SIMPAN MODEL PIPELINE KE FILE PICKLE
with open('eigenface_pipeline.pkl', 'wb') as f:
  pickle.dump(pipe, f)


#PIPELINE UNTUK LOAD MODEL YANG TELAH DITRAIN
pipe = pickle.load(open('eigenface_pipeline.pkl', 'rb'))


#MENAMPILKAN VISUALISASI EIGENFACE
# n_components = len(pipe[1].components_)

# ncol = 4
# nrow = (n_components + ncol - 1) // ncol

# n_plots = min(nrow * ncol, n_components)
# fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5*nrow),
# subplot_kw={'xticks':[], 'yticks':[]})

# eigenfaces = pipe[1].components_.reshape((n_components,
# X_train.shape[1]))
# for i, ax in enumerate(axes.flat[:n_plots]):
#   ax.imshow(eigenfaces[i].reshape(face_size), cmap='gray')
#   ax.set_title(f'Eigenface {i+1}')

# plt.tight_layout()
# plt.show()


#FUNGSI UNTUK MENGHITUNGKAN SKOR EIGENFACE
def get_eigenface_score(X):
  X_pca = pipe[:2].transform(X)
  eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1)
  return eigenface_scores


#FUNGSI UNTUK MELAKUKAN PREDIKSI GAMBAR WAJAH
def eigenface_prediction(image_gray):
  faces = detect_faces(image_gray)
  cropped_faces, selected_faces = crop_faces(image_gray, faces)

  #RETURN STRING JIKA TIDAK ADA WAJAH YANG TERDETEKSI
  if len(cropped_faces) == 0:
    return 'Tidak ada wajah terdeteksi'

  X_face = []
  for face in cropped_faces:
    face_flattened = resize_and_flatten(face)
    X_face.append(face_flattened)

  X_face = np.array(X_face)
  labels = pipe.predict(X_face)
  scores = get_eigenface_score(X_face)

  return scores, labels, selected_faces


#FUNGSI UNTUK MEMBENTUK BOUNDING BOX  DAN TULISAN PADA WAJAH TERDETEKSI
def draw_text(image, label, score,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(0, 0),
              font_scale=0.6,
              font_thickness=2,
              text_color=(0, 0, 0),
              text_color_bg=(0, 255, 0)
              ):

  x, y = pos
  score_text = f'Score: {score:.2f}'
  (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
  (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
  cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1)
  cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
  cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

#FUNGSI UNTUK MENAMPILKAN HASIL DETEKSI DAN REKOGNISI WAJAH
def draw_result(image, scores, labels, coords):
  result_image = image.copy()
  for (x, y, w, h), label, score in zip(coords, labels, scores):
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    draw_text(result_image, label, score, pos=(x, y))
  return result_image  

#MELAKUKAN DETEKSI DAN RECOGNISI DENGAN SAMPEL GAMBAR
sample_scores, sample_labels, sample_face = eigenface_prediction(sample_image_gray)
sample_result = draw_result(sample_image_rgb, sample_scores, sample_labels, sample_face)
plt.imshow(sample_result)

#FUNGSI UNTUK MELAKUKAN DETEKSI DAN REKOGNISI WAJAH SECARA REAL-TIME
def video_read():
    #INISIASI WEBCAM, 0 UNTUK WEBCAM BAWAAN, 1 UNTUK WEBCAM EKSTERNAL
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)

    #CEK KAMERA BISA DIBUKA ATAU TIDAK
    if not cap.isOpened():
        print("Tidak bisa membuka kamera")
        return

    print("Webcam bisa dibuka, pencet 'q' untuk keluar")

    #LOOP UNTUK MEMBACA FRAME DARI WEBCAM DAN MELAKUKAN DETEKSI WAJAH
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak bisa membaca frame dari kamera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #DETEKSI DAN REKOGNISI WAJAH
        prediction_result = eigenface_prediction(gray)
        result_frame = frame.copy()

        #MENAMPILKAN HASIL
        if isinstance(prediction_result, str):
            cv2.putText(result_frame, prediction_result, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            scores, labels, coords = prediction_result
            result_frame = draw_result(result_frame, scores, labels, coords)

        cv2.imshow('Real-Time Face Recognition', result_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_read()