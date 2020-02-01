![synthetic-disaster logo](img/logo.png)

Generates synthetic satellite images of natural disasters using deep neural networks.

![demo](img/demo.gif)


**Prerequisites**
---

+ [Docker](https://www.docker.com/)
+ [docker-compose](https://docs.docker.com/compose/)
+ [git lfs](https://git-lfs.github.com/)


**Installation**
---

```
$ git lfs pull
$ docker-compose build
```

![installation](img/installation.gif)


**Run Demo**
---

```
$ docker-compose up
```


**Test Samples**
---

| <img width=300> Input | <img width=300> Synthetic | <img width=300> Actual |
|--- | --- | --- |

<img src="img/test_samples/1.png" width="900">
<img src="img/test_samples/2.png" width="900">
<img src="img/test_samples/3.png" width="900">
<img src="img/test_samples/4.png" width="900">
<img src="img/test_samples/5.png" width="900">
<img src="img/test_samples/6.png" width="900">


**Acknowledgements**
---

 + [XView2 data-set](https://xview2.org/dataset)
 + [pix2pix paper](https://arxiv.org/abs/1611.07004)
 + [pix2pix implementation](https://github.com/znxlwm/pytorch-pix2pix)
