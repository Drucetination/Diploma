import cv2
import numpy as np
from pythonRLSA import rlsa


def get_roi(image):

    """ Возвращает координаты вершин прямоугольников, содержащих области интереса на изображении
        Аргумент image: ndarray - изображение
        Возвращает coordinates: list<list> - список списков формата [x0, x1, y0, y1]
                   roi: list<ndarray> - список изображений блоков контента
    """
    
    ret, bin_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    horizontal_smoothed_image = rlsa.rlsa(bin_image, True, False, 16)
    vertical_smoothed_image = rlsa.rlsa(bin_image, False, True, 8)

    smoothed_image = horizontal_smoothed_image & vertical_smoothed_image
    cv2.waitKey(0)

    ret01, inv_smoothed_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    dilation_kernel = np.ones((3, 3), dtype=int)

    dilated_image = cv2.dilate(inv_smoothed_image, dilation_kernel, iterations=2)

    ret2, labels = cv2.connectedComponents(dilated_image)

    roi = []
    roi_coordinates = []
    for label in range(1, ret2):
        area = np.where(labels == label)
        roi_coordinates.append([np.amin(area[1]), np.amax(area[1]), np.amin(area[0]), np.amax(area[0])])
        roi.append(image[np.amin(area[0]):np.amax(area[0]), np.amin(area[1]):np.amax(area[1])])

    return roi_coordinates, roi


