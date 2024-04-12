# MNIST
Elimizde farklı insanların rakam el yazılarından kesilmiş resimler bulunuyor ve bu resimlerin hangi rakamı temsil ettiğini biliyoruz. Amacımız, bu el yazılarını input olarak alıp karşılık gelen rakamı çıktı olarak belirleyen bir sinir ağı oluşturmak ve eğitmektir.

Bu amaçla, Python'da MNIST (Modified National Institute of Standards) veri kümesi gibi, el yazılarını piksel değerleriyle sayısallaştırılmış şekilde içeren bir veri kümesi bulunmaktadır. Bu veri kümesine erişmek için TensorFlow kütüphanesini kullanabiliriz. TensorFlow'u yüklemek için 'pip install tensorflow' komutunu kullanabilirsiniz.

Bu veri setini kullanarak, sinir ağını eğitebilir ve bir resmin hangi rakamı temsil ettiğini tahmin etmesini sağlayabiliriz.

# Yapay Sinir Ağları ile Rakam Tanıma
Projemizi beş adımda gerçekleştireceğiz. 
1. İş Problemi (Business Problem)
2. Veriyi Anlamak (Data Understanding)
3. Veriyi Hazırlamak (Data Preparation)
4. Modelleme (Modeling)
5. Değerlendirme (Evulation)
