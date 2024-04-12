import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

import matplotlib.pyplot as plt
import numpy as np

# Çalışma esnasındaki çeşitli uyarıları kapatmak için kullandım.
import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
filterwarnings('ignore')

# MNIST Veri Setinin Yüklenmesi
(x_train , y_train) , (x_test , y_test) = mnist.load_data()

# Eğitim Setinin Boyutuna Erişim
print("Eğitim Seti Boyutu : " , x_train.shape , y_train.shape)
# Eğitim Seti Boyutu :  (60000, 28, 28) (60000,)


# Test Setinin Boyutuna Erişim
print("Test Seti Boyutu : " , x_test.shape , y_test.shape)
# Test Seti Boyutu :  (10000, 28, 28) (10000,)


# Sınıf serisini bulmak için :
num_labels = len(np.unique(y_train))
# Burada hedef değişkenin sınıf sayısını aldık.


# VERİ SETİNDEN ÖRNEKLER GÖSTERİLMESİ

# Gösterilecek olan görsellerin 10a 10 olmasını istiyoruz.
plt.figure(figsize=(10,10))

#59000. veriyi alacağım mesela
plt.imshow(x_train[59000] , cmap='gray')

# 10 adet resimi bastırsın ekrana
plt.figure(figsize=(10,10))
for n in range(10) :
  ax = plt.subplot(5,5,n+1)               # 5 satır ve 5 sütun olsun ve bunları alt alta yazdırsın.
  plt.imshow(x_train[n] , cmap='gray')    # Okuma işlemini yapan koddur.
  plt.axis('off')                         # Bu kod eksenleri göstermez.
  plt.show()                              # Görselleri ekrana bastırır.

# Üstteki kodu fonksiyon olarak tanımlayalım.
def visualize_img(data) :
  plt.figure(figsize=(10,10))
  for n in range(10):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(data[n], cmap='gray')
    plt.axis('off')
    plt.show()


# Bilgisayar resimleri RGB formatta okur.
x_test[2]


sample_data = x_test[0]                   # Örnek veri değişkenine x_test'in ilk verisini atadık
plt.imshow(sample_data, cmap="gray" )      # Örnek veriyi ekrana bastırdık.


sample_data[7,10]     # Bu kodda sample_test[0][7,10] yazmadık çünkü sample_test tanımlaması zaten x_test[0] şeklindedir.
x_test[0][7,10]       # Bu kod bize 7. sütun ve 10. satırdaki rengin değerini tek olarak döndürür.


sample_data.sum()     # Veri üzerinde matematiksel işlemler yapabiliriz.
sample_data.mean()    # Verideki renk ortalamasını aldık.

x_test[0].sum()       # Yazım bu şekilde de olabilir.
x_test[0].mean()      # Yazım bu şekilde de olabilir.


 # Resimde bir yerde kesit alabiliriz. (7 sayısının şapkasını aldık)
sample_data[7:10 , 6:21]


# Resimde kesit aldığımız yerin ortalamasını alabiliriz.
cut_sample_data = sample_data[5:20,10:20]
cut_sample_data.mean()



# Verileri sayılarla görselleştirmek için şöyle bir fonksiyon kullanabiliriz.
def pixel_visualize(img) :
  fig = plt.figure(figsize=(12,12))
  ax = fig.add_subplot(111)
  ax.imshow(img, cmap="gray")
  width , height = img.shape

  threshold = img.max() / 2.5

  for x in range(width) :
    for y in range(height) :
      ax.annotate(str(round(img[x][y] , 2)) , xy=(y,x) , color='white' if img[x][y]<threshold else 'black')


pixel_visualize(sample_data)    # Fonksiyonu çağırdık.

# NOT : Şu ana kadar iş problemini tanımladık ve veriyi anladık.

# VERİYİ HAZIRLAMA
y_train[0:5]  # Bu kod bize y olduğu için çıktıdaki , train olduğu için eğitimdeki : Yani eğitim çıktılarındaki etiketleri verir.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Reshape
# Elimizdeki veriyi, yapay zekanın kendi yapısal formunda algılayabilmesi için yeniden şekillendirme işlemi.

image_size = x_train.shape[1]

print(f"x_traing boyutu :  {x_train.shape}")
print(f"x_test   boyutu :  {x_test.shape}")
# x_traing boyutu :  (60000, 28, 28)
# x_test   boyutu :  (10000, 28, 28)


x_train = x_train.reshape(x_train.shape[0] , 28,28,1)
x_test = x_test.reshape(x_test.shape[0] , 28,28,1)
print(f"x_traing boyutu :  {x_train.shape}")
print(f"x_test   boyutu :  {x_test.shape}")
# x_traing boyutu :  (60000, 28, 28, 1)
# x_test   boyutu :  (10000, 28, 28, 1)

# Normalizasyon
# Elimizdeki her bir pikselde yer alan değerleri belirli bir standarta dönüştürmüş olacağız.
# Her bir verideki 0-255 aralığındaki değerleri 0 ya da 1 değeri şekline çeviriyoruz. Neden Yapıyoruz ?
# - Eğitim süresinin daha hızlı olması için
# - Öğrenme sürecinin daha doğru bir şekilde ilerlemesi için


# 0 ile 1 arasına almanın en iyi yolu 255 sayısına bölmektir.
# float32 dönüşümü ile performans arttırdık.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# Modelleme
# Modelin iç (internal) ve dış (external) özelliklerini belirlememiz lazım.
# İlk dış özellikler ve ardından iç özelliklere değinebiliriz.

# External Features
model = tf.keras.Sequential([   # Bir model nesnesi oluşturduk. Sequential = Sıralı
    # Bu yapı sayesinde sıralı katmanlardan oluşan bir sinir ağı çok kolay oluşturulabilir.


  Flatten(input_shape=(28,28,1)),                                         # Biz reshape'den geçirdik ve boyut var dedik. Bunu sinir ağına da diyoruz.
  Dense(units=128, activation='relu', name='layer1'),                      # İlk katmanımız. Bir gizli katman. Bu gizli katman içerisinde 128 adet nöron var.
                                                                          # 128 adet nöron sayısı ve activation değişebilir. Bunlar bize bağlıdır.
                                                                          # 128 olmasının sebebi ise MNIST için en uygun 128 nöron olduğu yazıyormuş kaynaklarda


  Dense(units=num_labels, activation='softmax', name='output_layer')      # Bu kod ise gizli katmanın sonudur. 10 adet etiket tipi (çoklu veri) olduğundan softmax kullanılmıştır.
])                                                                        # İkili sınıflandırma olsaydı Sigmoid fonksiyonu olurdu.
                                       # Nöron sayısı da etiket sayısı kadar verilmiş. Tahmin süreci gerçekleştiğinde her bir sınıfa ait olma olsılıkları çıkmış olacak.

# Internal Features
model.compile(loss='categorical_crossentropy',                # compile = derleme
                                                              # loss ifadesi hata değerlendirme etiğidir. Amaç bu fonksiyonu minimize etmektir.
                                                              # Burada çoklu sınıflandırma olduğu için  'categorical_crossentropy' yöntemini kullandık.
                                                              # Bu bize optimize edilmesi gereken kayıp fonksiyonu ifade ediyor.

              optimizer='adam',                               # Optimize işlemini ise adam algoritması denenmiştir. Burada değiştirilebilir.



              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall() , "accuracy"]
              # accuracy = doğruluk , precision = hassas , recall = geri çağırma

              # Loss fonksiyonunu minimize etmemiz gerekiyordu. Peki bu minimize işleminde gidiş yönümüzü nasıl belirleriz?
              # Hata metriklerine bakarak belirleyebiliriz.

)


model.summary()

# Burada 784 ifadesi 28*28.
# 100480 ifadesi 784*128.
# 1290 ifadesi 128*10 + 10.   10 tane bias değeri eklendi.
"""

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 layer1 (Dense)              (None, 128)               100480

 output_layer (Dense)        (None, 10)                1290

=================================================================
Total params: 101770 (397.54 KB)
Trainable params: 101770 (397.54 KB)
Non-trainable params: 0 (0.00 Byte)

"""

# Total Parametre Sayısı
# Eğitilebilir Parametre Sayısı
# Eğitilemez Parametre Sayısı



# Modeli Eğitmek yani fit etmek için
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
#model.fit(girdi eğitim seti, çıktı eğitim seti, kaç adet optimizasyon işlemi yapsın(kaç tur dönsün),
# her turda 128 tane gözlem birine odaklanarak ilgili optimizasyon yöntemine dayalı olarak gradyan türevi hesaplanıp bir sonraki epoch'a geçicek , doğrulama verileri  )

# VALİDATİON = DOĞRULAMA



# DEĞERLENDİRME 
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

###################################
### ACCURACY VE LOSS GRAFİKLERİ ###
###################################

#------- GRAFİK 1 ACCURACY -------#

plt.figure(figsize=(20,5))            # Boyut ayarlaması yapıldı.
plt.subplot(1,2,1)                    # Birden fazla grafik kullanacağımızdan bunun bilgisini verdik. Bire ikilik ve birinci grafik bu diye.
plt.plot(history.history['accuracy'], color='b', label='Training Accuracy')          # Accuracy değerini yazdırıyoruz.
plt.plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')    # Validation Accuracy değerini yazdırıyoruz.

# İsimlendirme Ayarları Yapıyoruz
plt.legend(loc='lower right')
plt.xlabel('Epoch' , fontsize=16)
plt.ylabel('Accuracy' , fontsize=16)

# Y eksenindeki limitleri belirliyoruz.
plt.ylim([min(plt.ylim()), 1])
plt.title('Eğitim ve Test Başarım Grafiği' , fontsize=16)


#--------- GRAFİK 2 LOSS ---------#

plt.subplot(1,2,2)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch' , fontsize=16)
plt.ylabel('Loss' , fontsize=16)
plt.ylim([0 ,max(plt.ylim())])
plt.title('Eğitim ve Test Kayıp Grafiği' , fontsize=16)
plt.show()


loss, precision, recall, acc = model.evaluate(x_test, y_test, verbose=False)
print("\nTest Accuracy : %.1f %%" % (100.0 * acc))
print("\nTest Loss : %.1f %%" % (100.0 * loss))
print("\nTest Precision : %.1f %%" % (100.0 * precision))
print("\nTest Recall : %.1f %%" % (100.0 * recall))


# Modeli tahmin için kullanılmak istenirse, README.md içerisinde örneğe yer verilmiştir.