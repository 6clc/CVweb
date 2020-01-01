import cv2
import  scipy.ndimage
import numpy


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]
    # print(predictions_shape, shape_r, shape_c)

    pred = pred / numpy.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    img = img / numpy.max(img) * 255

    return img


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os
    img = cv2.imread('/home/he/Data/DataSets/SALICON/maps/val/COCO_val2014_000000000164.png', 0)
    print(type(img))
    plt.imshow(img, cmap='gray')
    plt.show()
    shape_r, shape_c = img.shape
    img = postprocess_predictions(img, shape_r, shape_c)
    img = img.astype(numpy.uint8)
    plt.imshow(img, cmap='gray')
    plt.show()