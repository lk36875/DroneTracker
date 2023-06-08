## Wykrywanie i śledzenie dronów

Proces wykrywania dronów został podzielony na dwa etapy - detekcję obiektów oraz ich śledzenie. Detekcja zajmuje się wykrywaniem obiektów na pojedynczych klatkach filmowych, które są traktowane jak niezależne zdjęcia. 

Dane o wykrytych obiektach są następnie przekazywane do odpowiednich metod śledzenia, które są w stanie stwierdzić, czy wykryty obiekt jest tym samym, który został wykryty kilka klatek wcześniej. Do wykrywania dronów użyta została sieć neuronowa YOLO (You Only Look Once), która jest jedną z najbardziej popularnych metod detekcji obiektów w czasie rzeczywistym. Wagi modelu YOLO, które zostały wyuczone na dużym zbiorze danych COCO. Następnie przeprowadzone zostały dalsze treningi, aby dostosować model do wykrywania dronów. 

W przypadku śledzenia, użytkownik ma możliwość skorzystania z jednej z sześciu metod:
• metody śledzenia SORT, DeepSORT
• metody śledzenia obiektu po cechach i wzorcach poruszania się takie jak CSRT, MEDIANFLOW, KCF
• wykrywanie dronów bazując na optycznym przepływie (optical flow) i identyfikacja wykrytych obiektów


Zwracany jest film z zaznaczonym dronem oraz wykryte bounding boxy.

![Drone](/images/drone-image.png)

W projekcie użyto repozytoria:
<li>
https://github.com/nwojke/deep_sort
<li>
https://github.com/abewley/sort