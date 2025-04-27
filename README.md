# 👤 Face Detection and Recognition

_Deteksi dan Rekognisi Wajah_

**Nama:** Muhammad Hanif Zuhair  
**NIM:** 23/516550/PA/22099

---

## 🚀 **Petunjuk Menjalankan Program**

Untuk memulai dan menjalankan program deteksi dan rekognisi wajah, ikuti langkah-langkah berikut dengan seksama:

1.  **Siapkan Semua File**

    Download kode beserta file pendukung lainnya, lalu letakkan dalam satu folder (sebagai folder utama proyek).

2.  **Download Dataset**

    Download dataset yang digunakan melalui link Google Drive yang sudah diisikan pada form, lalu letakkan file dataset dalam folder yang sama dengan sebelumnya.

3.  **Ekstrak Dataset**

    Ekstrak file `archive.zip`, yang berisikan dataset yang akan digunakan. Setelah ekstraksi, pastikan folder `images` muncul di dalam folder proyek.

4.  **Siapkan Virtual Environment**

    Buat virtual environment dan pastikan sudah terinstal dan teraktivasi dengan benar. Jika belum terinstall, bisa jalankan kode berikut:

    1.  **Install Virtual Environment**  
        Jika Anda belum menginstall virtual environment, jalankan perintah ini di terminal:

         ```bash
         pip install virtualenv
         ```

    2.  **Buat Virtual Environment dan Aktivasi**  
        Masuk ke folder proyek terlebih dulu dan kemudian buat virtual environment dan aktivasi dengan perintah berikut:

        ```bash
        python -m venv .venv
        ```

         ```bash
         .venv\Scripts\activate
         ```

5.  **Install Library yang Dibutuhkan**  
    Pastikan sudah menginstall semua library yang diperlukan. Apabila belum, jalankan perintah berikut:
    ```bash
    pip install opencv-python numpy matplotlib scikit-learn
    ```

6.  **Modifikasi Kode `main.py`**  
    Buka file `main.py` dan cari baris kode berikut:
    ```py
    cap = cv2.VideoCapture(1)
    ```
    Ubah angka 1 menjadi 0 jika Anda menggunakan webcam bawaan laptop. Jika menggunakan webcam eksternal, biarkan seperti semula.

7.  **Jalankan Program**  
    Untuk dapat menjalankan program, jalankan perintah berikut:
    ```bash
    py main.py
    ```

## 📸 Dokumentasi Hasil Program

(Tambahkan screenshot atau hasil dari program di sini untuk menunjukkan betapa kerennya hasilnya!)
Untuk menjalankan program, ikuti langkah-langkah berikut:


https://github.com/user-attachments/assets/5a34c203-5e61-458a-a448-169d93573299

