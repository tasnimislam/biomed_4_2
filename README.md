# Full dataset:
https://www.kaggle.com/datasets/tasnimnishatislam/fingerprint-data-tushar

https://l.messenger.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1kCRso9Lk6JhDO92OES3GIjKuEzYb0LQq%3Fusp%3Dshare_link&h=AT2CGrQaqJOg2q3UC4DhpXFiKAFjS5G3nerkU-9wfUVqVIRFvYOvRC4eI7EN7KgmM_iIseNGCYtcRAjGMKMKfzJywpI7Yb9tNkVNvUiQd1514H4eOkKXvJum8VVkexS267876Q

https://l.messenger.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1kCRso9Lk6JhDO92OES3GIjKuEzYb0LQq%3Fusp%3Dshare_link&h=AT2CGrQaqJOg2q3UC4DhpXFiKAFjS5G3nerkU-9wfUVqVIRFvYOvRC4eI7EN7KgmM_iIseNGCYtcRAjGMKMKfzJywpI7Yb9tNkVNvUiQd1514H4eOkKXvJum8VVkexS267876Q


# Kaggle links:

https://www.kaggle.com/code/tasnimnishatislam/fingerprint-to-blood-group-full-pipeline/notebook

https://www.kaggle.com/code/tasnimnishatislam/image-processing-fingerprint

# 21/01/2016

Current data: 37

Current accuracy: 0.375 (highest), tried individual 4 type of images

Current class numbers: 4

Flactuates a lot due to poor model structure

Getting same class in prediction

Possible reasons:

1. Model too complex, overfitiing

2. Imbalanced data: distribution: 15(o pos), 11(b pos), 7(a pos), 2 (ab pos)

3. Optimizer, regualrizer, dropout, activation layer focus

4. Tried simple MLP, vgg net on top plus mlp, resnet

5. Image processing issue

Next step

1. Image processing: follow https://www.sciencedirect.com/science/article/pii/S0045790621003554?via%3Dihub

2. Combine 4 images, do patch wise classification, apply cutmix to generate many data

3. More Ab pos data collect and try balnace imbalance
