# Audio-classification
Классификация аудио с помощью Python


Звук представлен в форме аудиосигнала с такими параметрами, как частота, полоса пропускания, децибел и т.д. Типичный аудиосигнал можно выразить в качестве функции амплитуды и времени.


 Из  спектрограмм я провела анализ аудиоданных и извлекла  характеристики в виде среднего, дисперсии и др. значений с помощью библиотеки  librosa.   Для классификации  “живого” голоса (класс 1) и его отделению от синтетического/конвертированного/перезаписанного голоса (класс 2) я  использовала ML алгоритм  SVM (Support Vector Machines) / машины опорных векторов.

SVM работает путем сопоставления данных с многомерным пространством функций, чтобы точки данных можно было классифицировать, даже если данные не могут быть линейно разделены иным образом. 

Для работы я использовала математическую функцию, используемой для преобразования (известна как функция ядра) -  RBF (радиальную базисную функцию).
Результат: разработала модель классификации; точность классификатора составляет: Train set Accuracy:  0.979725
Test set Accuracy:  0.9713

**Мел-частотные кепстральные коэффициенты (MFCC)**

Представляют собой  набор признаков , которые  описывают общую форму спектральной огибающей. Они моделируют характеристики человеческого голоса. MFCC -  ****коэффициенты частотной капсулы, суммируют частотное распределение по размеру окна. Поэтому можно анализировать как частотные, так и временные характеристики звука.


# **Cпектрограмма**

Спектрограмма — это визуальный способ представления уровня или “громкости” сигнала во времени на различных частотах, присутствующих в форме волны. Обычно изображается в виде [тепловой карты](https://ru.wikipedia.org/wiki/%D0%A2%D0%B5%D0%BF%D0%BB%D0%BE%D0%B2%D0%B0%D1%8F_%D0%BA%D0%B0%D1%80%D1%82%D0%B0).

`.stft()` преобразует данные в кратковременное преобразование Фурье. С помощью STFT можно определить амплитуду различных частот, воспроизводимых в данный момент времени аудиосигнала.





