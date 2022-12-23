import numpy as np
import cv2


def check_value(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if not((img[i][j] == 0) or (img[i][j] == 255)):
                return False
    return True


predicted = cv2.imread('u2net/u2net_bce_itr_5000_train_0.745451_tar_0.047403/33.png', cv2.IMREAD_GRAYSCALE)
ret,predicted_prime = cv2.threshold(predicted,100,255,cv2.THRESH_BINARY_INV)
ret,predicted = cv2.threshold(predicted,100,255,cv2.THRESH_BINARY)
groundtruth = cv2.imread('u2net/groundtruth/33.jpg', cv2.IMREAD_GRAYSCALE)
ret,groundtruth_prime = cv2.threshold(groundtruth,100,255,cv2.THRESH_BINARY_INV)
ret,groundtruth = cv2.threshold(groundtruth,100,255,cv2.THRESH_BINARY)

print(f'predicted: {check_value(predicted)}')
print(f'predicted_prime: {check_value(predicted_prime)}')
print(f'groundtruth: {check_value(groundtruth)}')
print(f'groundtruth_prime: {check_value(groundtruth_prime)}')


True_Positive = np.logical_and(predicted, groundtruth)
True_Negative = np.logical_and(predicted_prime, groundtruth_prime)
False_Positive = np.logical_and(predicted, groundtruth_prime)
False_Negative = np.logical_and(predicted_prime, groundtruth)

True_Positive = sum(sum(True_Positive))
True_Negative = sum(sum(True_Negative))
False_Positive = sum(sum(False_Positive))
False_Negative = sum(sum(False_Negative))

print(f'True_Positive: {True_Positive}')
print(f'True_Negative: {True_Negative}')
print(f'False_Positive: {False_Positive}')
print(f'False_Negative: {False_Negative}')

Precision = True_Positive / (True_Positive + False_Positive)
Recall = True_Positive / (True_Positive + False_Negative)
F1 = (2*Recall*Precision) / (Recall + Precision)

print(f'Precision: {Precision*100}')
print(f'Recall: {Recall*100}')
print(f'F1: {F1*100}')


subscription = np.logical_and(predicted, groundtruth)
aggregation = np.logical_or(predicted, groundtruth)

subscription = sum(sum(subscription))
aggregation = sum(sum(aggregation))
predicted = sum(sum(predicted))
groundtruth = sum(sum(groundtruth))

print(f'subscription: {subscription}')
print(f'aggregation: {aggregation}')
print(f'predicted: {predicted}')
print(f'groundtruth: {groundtruth}')

Dice = ((2*subscription)/(predicted + groundtruth)) * 100
Jaccard = (subscription/aggregation) * 100
Dice = ((2*True_Positive)/((2*True_Positive) + False_Positive + False_Negative)) * 100
Jaccard = (True_Positive/(True_Positive + False_Positive + False_Negative)) * 100

print(f'Dice: {Dice}')
print(f'Jaccard: {Jaccard}')
