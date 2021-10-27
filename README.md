# Audio-classification
Классификация аудио с помощью Python
Классификация аудио  с помощью python

Звук представлен в форме аудиосигнала с такими параметрами, как частота, полоса пропускания, децибел и т.д. Типичный аудиосигнал можно выразить в качестве функции амплитуды и времени.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a6d762f7-7e30-4cfd-9daa-b7e141eec612/Untitled.png)

 Из  спектрограмм я провела анализ аудиоданных и извлекла  характеристики в виде среднего, дисперсии и др. значений с помощью библиотеки  librosa.   Для классификации  “живого” голоса (класс 1) и его отделению от синтетического/конвертированного/перезаписанного голоса (класс 2) я  использовала ML алгоритм  SVM (Support Vector Machines) / машины опорных векторов.

SVM работает путем сопоставления данных с многомерным пространством функций, чтобы точки данных можно было классифицировать, даже если данные не могут быть линейно разделены иным образом. 

Для работы я использовала математическую функцию, используемой для преобразования (известна как функция ядра) -  RBF (радиальную базисную функцию).
Результат: разработала модель классификации; точность классификатора составляет: Train set Accuracy:  0.979725
Test set Accuracy:  0.9713

график управления амплитудой формы волны:

```python
import IPython.display as ipd
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)
ipd.Audio(audio_data)
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d5689369-1176-425d-8018-462818b90bab/Untitled.png)

**Мел-частотные кепстральные коэффициенты (MFCC)**

Представляют собой  набор признаков , которые  описывают общую форму спектральной огибающей. Они моделируют характеристики человеческого голоса. MFCC -  ****коэффициенты частотной капсулы, суммируют частотное распределение по размеру окна. Поэтому можно анализировать как частотные, так и временные характеристики звука.

```python
# Calculate MFCCs
mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)
plt.figure(figsize=(15, 5))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/44d00748-d10e-4cff-bad4-c53bdaeded44/Untitled.png)

```python
mfccs
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31c71646-5ad4-4638-a782-ab5cc941ee47/Untitled.png)

# **Cпектрограмма**

Спектрограмма — это визуальный способ представления уровня или “громкости” сигнала во времени на различных частотах, присутствующих в форме волны. Обычно изображается в виде [тепловой карты](https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D0%BF%D0%BB%D0%BE%D0%B2%D0%B0%D1%8F_%D0%BA%D0%B0%D1%80%D1%82%D0%B0).

`.stft()` преобразует данные в кратковременное преобразование Фурье. С помощью STFT можно определить амплитуду различных частот, воспроизводимых в данный момент времени аудиосигнала.

```python
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4747381c-3409-4d9a-b891-73396f159397/Untitled.png)

```python
print("Prediction:", yhat[0:20])
print("Real Value:", y_test[0:20])
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bd0dfbae-29f5-4015-809c-d04f2285a2ec/Untitled.png)

```python
# accuracy evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, clf.predict(X_train)))
print("Test set Accuracy: ",metrics.accuracy_score(y_test, yhat) )
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c19c2416-2d58-4e2d-a19f-55891c959cf1/Untitled.png)

```python
from sklearn.metrics import classification_report,confusion_matrix

print('CONFUSION_MATRIX :\n')
print(confusion_matrix(y_test,yhat))
print('\n')
print('REPORT :\n')
print(classification_report(y_test,yhat))
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38c7f0ca-30b7-4092-a749-4ae22b4d3811/Untitled.png)
