#selecting data for model
"""
-data set memiliki terlalu banyak variabel untuk membungkus head data 

-bagimana caranya mengurangi data tersebut sehingga menjadi data yg mudah untuk dipahami?
=>memilih beberapa variabel
=>menggunakan teknik statistik untuk memproitaskan sebuah variabel secara otomatis

-untuk memilih sebuah variabel/kolom, diperlukan:
=>melihat daftar semua kolom dalam kumpulan data
    =>poperti kolom dari data frame
"""
import pandas as pd
file_path="intro to ml\melb_data.csv"
melbroune_data=pd.read_csv(file_path)
print(melbroune_data.columns)
print(melbroune_data)
"""
Melbroune_data memiliki missing values(beberapa rumah untuk variabelnya tidak di recorded)
# Kita akan belajar menangani nilai yang hilang di tutorial selanjutnya.
# Data Iowa Anda tidak memiliki nilai yang hilang di kolom yang Anda gunakan.
# Jadi kami akan mengambil opsi paling sederhana untuk saat ini, dan menghapus rumah dari data kami.
# Jangan terlalu khawatir tentang ini untuk saat ini, meskipun kodenya adalah:
# dropna menjatuhkan nilai yang hilang (anggap saja sebagai "tidak tersedia")
"""
melbroune_data=melbroune_data.dropna(axis=0)#data yang hilang (nan) pada baris(axis=1) dihilangkan satu baris
print(melbroune_data)
"""
There are many ways to select a subset of your data. The Pandas course covers these in more depth, but we will focus on two approaches for now.

=> Dot notation, which we use to select the "prediction target"
=> Selecting with a column list, which we use to select the "features"
"""

#Selecting The Prediction target
"""
Anda dapat mengeluarkan variabel dengan notasi titik. 
Kolom tunggal ini disimpan dalam Seri, yang secara umum seperti DataFrame dengan hanya satu kolom data.
Kami akan menggunakan notasi titik untuk memilih kolom yang ingin kami prediksi, 
yang disebut target prediksi. Dengan konvensi, target prediksi disebut y.
Jadi kode yang kita perlukan untuk menyimpan harga rumah di data Melbourne adalah
"""
y=melbroune_data.Price
print(y)

#chossing features
"""
Kolom yang dimasukkan ke dalam model kita (dan kemudian digunakan untuk membuat prediksi) disebut "fitur".
Dalam kasus kami, itu akan menjadi kolom yang digunakan untuk menentukan harga rumah. Terkadang, 
Anda akan menggunakan semua kolom kecuali target sebagai fitur. Di lain waktu Anda akan lebih baik dengan lebih sedikit fitur.
Untuk saat ini, kami akan membuat model dengan hanya beberapa fitur. 
Nanti Anda akan melihat cara mengulangi dan membandingkan model yang dibuat dengan fitur berbeda.
Kami memilih beberapa fitur dengan memberikan daftar nama kolom di dalam tanda kurung.
Setiap item dalam daftar itu harus berupa string (dengan tanda kutip).

Berikut ini contohnya:
"""
melbroune_features = [ "Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
"""
secara konvensi data tersbeut dapat dipanggil dengan data x
"""
X=melbroune_data[melbroune_features]

"""
Mari dengan cepat meninjau data yang akan kita gunakan untuk memprediksi harga rumah
menggunakan metode deskripsikan dan metode kepala, yang menunjukkan beberapa baris teratas.
"""
nilai=X.describe()
print(nilai)
head=X.head()
print(head)

#building the model
from sklearn.tree import DecisionTreeRegressor
"""
Langkah-langkah untuk membangun dan menggunakan model adalah:

-define
What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too

-fit
Menangkap pola dari data yang disediakan. Ini adalah jantung dari pemodelan.

-predict


-evaluate
menentukan seberapa akurat modle yang ditentukan

contoh:=>model
descision tree dari sklearn scikit-learn and fitting yang di sesuaikan berdasarkan
prdection target dan fitur
"""
## Tentukan model. Tentukan nomor untuk random_state untuk memastikan hasil yang sama setiap kali dijalankan
melbroune_model = DecisionTreeRegressor(random_state=1)

#fit model
melbroune_model= melbroune_model.fit(X, y)
print(melbroune_model)

"""
Banyak model pembelajaran mesin memungkinkan beberapa keacakan dalam pelatihan model.
Menentukan nomor untuk random_state memastikan Anda mendapatkan hasil yang sama di setiap proses. Ini dianggap sebagai praktik yang baik.
Anda menggunakan nomor apa pun, dan kualitas model tidak akan sepenuhnya bergantung pada nilai yang Anda pilih.
Kami sekarang memiliki model pas yang dapat kami gunakan untuk membuat prediksi.
Dalam praktiknya, Anda sebaiknya membuat prediksi untuk rumah baru yang akan datang di pasar daripada rumah yang sudah kami hargai.
Namun kita akan membuat prediksi untuk beberapa baris pertama dari data pelatihan untuk melihat cara kerja fungsi prediksi.

data di prediksi (y->prediction target) berdasarkan faktor yang mempengaruhi (x->fitur)
"""
print("Membuat Prediksi untuk 5 rumah berikut :\n{}".format(X.head()))
print("Thre Prediction:\n")#hanya memprediksi berdasarkan 5 baris data dengan 5 faktor fitur yang mempengauhi prediction target(Price rumah)
print("=> ",melbroune_model.predict(X.head()))