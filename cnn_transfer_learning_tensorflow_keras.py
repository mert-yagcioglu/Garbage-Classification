#MERT YAĞCIOĞLU 
#Kağıt ve Cam görüntülerini sınıflandırma.

import tensorflow

# VGG16 önceden eğitilmiş bir CNN modelini kullanıyoruz.
conv_base = tensorflow.keras.applications.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )

# Katmanları gösteriyoruz.
conv_base.summary()

#Hangi katmanların eğitilip dondurulacağına karar vermemiz gerekiyor. Çok fazla katman var.
#'block5_conv1' donduruluncaya kadar devam et diyoruz.
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#Boş model oluşturuyoruz.
model = tensorflow.keras.models.Sequential()

#Başta bahsettiğimiz VGG16 modelini concolutional katman olarak ekliyoruz. Bu katman fotoğrafın üzerinde geziniyor
model.add(conv_base)

#Katmanlarımız matrislerden vektöre dönüştürülüyor.
model.add(tensorflow.keras.layers.Flatten())

#Nöral katmanımızı ekliyoruz
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-5), #EN:in TensorFlow 2.x, the lr argument has been renamed to learning_rate         TR:Yeni sürüm ile lr yerine learning_rate olmuş
              metrics=['acc'])

#Oluşturduğumuz model
model.summary()

#Verilerin bulunduğu konumu gösteriyoruz.
train_dir = r'C:\Users\myyag\OneDrive\Masaüstü\BM\data\train'
validation_dir = r'C:\Users\myyag\OneDrive\Masaüstü\BMdata\validation'
test_dir = r'C:\Users\myyag\OneDrive\Masaüstü\BM\data\test'

#Overfitting yani aşırı öğrenme sorununun önüne geçmemiz için veri arttırımı yani görüntü arttırımı yapıyoruz.
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255, # piksel değerleri 0-255'den 0-1 arasına getiriliyor.
      rotation_range=40, # istenilen artırma işlemleri yapılabilir.
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        )

#İstendiği takdirde görüntü arttırma teknikleri uygulanabilir(360 derece çevirme, gürültü ekleme, grilik ekleme, döndürme vs.). Burada artırılmış görüntülere ihtiyacımız yok.
validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        )

#Modelimizi eğitiyoruz.
history = model.fit( #fit_generator yerine fit olmuş
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=5)

#Eğitilmiş modelimizi kaydedelim.
model.save('trained_tf_model.h5')

#Yukarıdaki validation kısmında yaptığımız gibi test kısmında da arttırılmış görüntülere ihtiyacımız yok.
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        )

#Test sonuçlarını yazdıralım.
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
