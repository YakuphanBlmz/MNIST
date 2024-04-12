![image](https://github.com/YakuphanBlmz/MNIST/assets/106194461/125d86b3-f158-4bb8-b158-62f83c3d374b)# MNIST
Elimizde farklı insanların rakam el yazılarından kesilmiş resimler bulunuyor ve bu resimlerin hangi rakamı temsil ettiğini biliyoruz. Amacımız, bu el yazılarını input olarak alıp karşılık gelen rakamı çıktı olarak belirleyen bir sinir ağı oluşturmak ve eğitmektir.

Bu amaçla, Python'da MNIST (Modified National Institute of Standards) veri kümesi gibi, el yazılarını piksel değerleriyle sayısallaştırılmış şekilde içeren bir veri kümesi bulunmaktadır. Bu veri kümesine erişmek için TensorFlow kütüphanesini kullanabiliriz. TensorFlow'u yüklemek için ``'pip install tensorflow``` komutunu kullanabilirsiniz.

Bu veri setini kullanarak, sinir ağını eğitebilir ve bir resmin hangi rakamı temsil ettiğini tahmin etmesini sağlayabiliriz.

<br>

# Yapay Sinir Ağları ile Rakam Tanıma
Projemizi beş adımda gerçekleştireceğiz. 
**1.** İş Problemi (Business Problem)
**2.** Veriyi Anlamak (Data Understanding)
**3.** Veriyi Hazırlamak (Data Preparation)
**4.** Modelleme (Modeling)
**5.** Değerlendirme (Evulation)

**NOT :** Kod içerisinde yorum satırlarında yeteri kadar Türkçe dili ile açıklama mevcuttur.

### Bağımlı/Bağımsız Değişkenler

**a. Bağımlı Değişken (Dependent Variable):**
Bağımlı değişken, modelin öğrenmeye çalıştığı veya tahmin etmeye çalıştığı ana hedef değişkendir. Bu değişken, diğer değişkenler tarafından etkilenebilir ve bu nedenle bağımsız değişkenlere bağlıdır. Model, bağımlı değişkenin değerini belirlemeye çalışır.

**b. Bağımsız Değişken (Independent Variable):**
Bağımsız değişkenler, bağımlı değişken üzerinde potansiyel bir etkiye sahip olan giriş değişkenleridir. Yapay zeka modelleri, bağımsız değişkenlerin değerlerini kullanarak bağımlı değişkenin tahminini yapmaya çalışır.

**NOT :** Doğru bağımsız değişkenlerin seçilmesi ve modelin bu değişkenleri kullanarak bağımlı değişkeni doğru bir şekilde tahmin etmesi önemlidir. 

**ÖRNEK**
- Bağımlı Değişken: Ev fiyatı
- Bağımsız Değişkenler: Yatak odası sayısı, banyo sayısı, evin konumu, evin büyüklüğü gibi faktörler

<br>

**ÇIKTI**
Epoch 1/5 469/469 [==============================] - 4s 6ms/step - loss: 0.3578 - precision_1: 0.9477 - recall_1: 0.8497 - accuracy: 0.9000 - val_loss: 0.1983 - val_precision_1: 0.9553 - val_recall_1: 0.9278 - val_accuracy: 0.9418 <br>
Epoch 2/5 469/469 [==============================] - 3s 5ms/step - loss: 0.1642 - precision_1: 0.9637 - recall_1: 0.9428 - accuracy: 0.9526 - val_loss: 0.1385 - val_precision_1: 0.9676 - val_recall_1: 0.9505 - val_accuracy: 0.9592 <br>
Epoch 3/5 469/469 [==============================] - 5s 11ms/step - loss: 0.1175 - precision_1: 0.9738 - recall_1: 0.9599 - accuracy: 0.9666 - val_loss: 0.1144 - val_precision_1: 0.9719 - val_recall_1: 0.9601 - val_accuracy: 0.9652 <br>
Epoch 4/5 469/469 [==============================] - 6s 12ms/step - loss: 0.0913 - precision_1: 0.9785 - recall_1: 0.9690 - accuracy: 0.9735 - val_loss: 0.0920 - val_precision_1: 0.9768 - val_recall_1: 0.9671 - val_accuracy: 0.9714 <br>
Epoch 5/5 469/469 [==============================] - 5s 11ms/step - loss: 0.0734 - precision_1: 0.9818 - recall_1: 0.9744 - accuracy: 0.9782 - val_loss: 0.0894 - val_precision_1: 0.9767 - val_recall_1: 0.9680 - val_accuracy: 0.9721 <br>
<keras.src.callbacks.History at 0x7ec6cebb4520>

**Bu değerleri inceleyecek olursak :**

**1.** accuracy yani doğruluk değeri , val_accrucay değeri ile benzer.
**2.** Loss değeri gitgide düşmüş. Ne kadar düşükse o kadar iyi.
**3.** accuracy: 0.9782 ise %97'lik bir başarı oranımız var demektir.

Eğer epochs değerini 10 yaparsak daha çok öğrenme olmuş olacaktır.

<br>

**Son Düzeltmeler ile Son Değerler :**
- Test Accuracy : 98.0 %
- Test Loss : 8.1 %
- Test Precision : 98.1 %
- Test Recall : 97.9 %

**NOT:** Accuracy, recall ve precision değerlerinin yakın çıkması demek, sınıflar arası bir dengesizlik olmadığı anlamına geliyor.

**a. Precision :** Mesela bizim 1 olarak tahmin ettiğimiz sınıfların, ne kadar 1 olduğunu kontrol eder. Tahmin ettiklerimizin başarısıdır. Yani hassaslığa bakar. Precision = Hassas<br>
**b. Recall :** Önce gerçek değerlere odaklanırız. Sonrasında biz bunların kaç tanesini doğru tahmin ettik diye bakıyoruz.

<br>
![image](https://github.com/YakuphanBlmz/MNIST/assets/106194461/46d367f0-f458-4138-8696-8f07b15006bc)
<br>

# Modelin Kayıt Edilmesi ve Tahmin İçin Kullanılması 

```model.save('mnist_model.h5')```                         => h5 dosya formatında kayıt gerçekleşir.
```random = random.randint(0, x_test.shape[0])```          => Bir adet örnek seçtik.
```test_image = x_test[random]```                          => Değişkene atadık.
```y_test[random]```                                       => Şimdi bakalım biz hangi etiketli veriyi almışız. Verinin etiketine bakalım.
```plt.imshow(test_image.reshape(28,28), cmap="gray")```   => İstersek bir de ekranda görelim.

![image](https://github.com/YakuphanBlmz/MNIST/assets/106194461/fbd1697d-079d-4ace-9ddc-63b8b7584ebc)



```test_data = x_test[random].reshape(1,28,28,1)```        => Önce reshape yapıyoruz modele sormak için.
```probability = model.predict(test_data)```               => Şimdi ise modelin tahmin etmesi için gereken kodu yazıyoruz.
```predict_classes = np.argmax(probability)```             => probability değişkeninde çıkan olasılık sonuçlarımız var. Bu olasılık değerleri arasından en yüksek olan değeri bulmamız gerekecek. Bunun için bu kodu yazabiliriz.
```print("Random Seçilen Sayı : " , predict_classes)```    => "Random Seçilen Sayı :  9"

**- Daha Ayrıntılı Bilgi İstersek :**

```print(f"Tahmin Edilen Sınıf:  {predict_classes} \n" )```
```print(f"Tahmin Edilen Sınıfın Olasılık Değeri:  {(np.max(probability, axis=-1))[0]} \n" )```
```print(f"Diğer Sınıfların Olasılık Değerleri: \n {probability} " )```

<br>
"
Tahmin Edilen Sınıf:  9 

Tahmin Edilen Sınıfın Olasılık Değeri:  0.9999940395355225 

Diğer Sınıfların Olasılık Değerleri: 
 [[1.5927174e-09 6.4466547e-15 2.9438849e-07 1.0376165e-07 3.1133129e-06
  1.5393294e-07 3.4492955e-12 2.3353400e-06 3.9469394e-10 9.9999404e-01]]
  
"












