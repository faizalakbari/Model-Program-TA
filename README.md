# Model-Program-TA

Terdapat 2 Program yang bisa dijalankan :

1) Model Deteksi : Program untuk klasifikasi model dengan tahapan Denoising,Ekstraksi Fitur dan Klasifikasi. Dataset menggunakan sinyal PPG dari subyek sehat dan subyek                    pasien CHD

2) Program Metode Seleksi Fitur : Program untuk memilih fitur dari dataset fitur hasil ekstraksi fitur dari Model Deteksi. Hasilnya adalah fitur-fitur yang terpilih yang                                   nantinya dapat digunakan sebagai referensi pada Model Deteksi saat memilih fitur.  Terdapat 3 Metode Seleksi Fitur yaitu Analysis of                                     Variance (ANOVA), Pearson Correlation dan Recursive Feature Elimination (RFE).


Cara Menggunakan Program Model Deteksi :

1) Download folder Model Deteksi
2) Buka file model_deteksi_ppg.py menggunakan editor python (Misal : Visual Studio Code)
3) Run Program
4) Pilih fitur yang akan digunakan
5) Klik tombol Run Klasifikasi
6) Hasil performa dapat dilihat pada console terminal

Cara Menggunakan Program Metode Seleksi Fitur :
1) Download folder Program Metode Seleksi Fitur
2) Buka file Algoritma_Metode_Seleksi_Fitur.ipynb menggunakan editor Jupyter atau Google Colab
3) Masukan dataset Feature_Butterworth.csv yang terdapat pada folder dataset
4) Run Program
5) Hasil dari fitur yang terpilih dapat dilihat pada section masing-masing metode (Anova, Pearson Correlation, RFE)
6) Gunakan fitur terpilih pada program Model Deteksi apabila ingin melakukan pengujian berdasarkan masing-masing metode seleksi fitur
