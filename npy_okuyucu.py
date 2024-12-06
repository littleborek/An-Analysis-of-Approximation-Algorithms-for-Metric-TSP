import numpy as np
import csv


npy_data = np.load('dataset_2_coordinates.npy')


with open('veri2.csv', 'w', newline='') as f:
    writer = csv.writer(f)


    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:

        writer.writerow(npy_data)

npy_data = np.load('dataset_3_coordinates.npy')


with open('veri3.csv', 'w', newline='') as f:
    writer = csv.writer(f)


    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:

        writer.writerow(npy_data)

npy_data = np.load('dataset_4_coordinates.npy')


with open('veri4.csv', 'w', newline='') as f:
    writer = csv.writer(f)


    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:
        writer.writerow(npy_data)

npy_data = np.load('dataset_5_coordinates.npy')

with open('veri5.csv', 'w', newline='') as f:
    writer = csv.writer(f)


    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:

        writer.writerow(npy_data)

npy_data = np.load('dataset_6_coordinates.npy')


with open('veri6.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:
        writer.writerow(npy_data)

npy_data = np.load('dataset_7_coordinates.npy')


with open('veri7.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:
        writer.writerow(npy_data)

npy_data = np.load('dataset_8_coordinates.npy')


with open('veri8.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:

        writer.writerow(npy_data)

npy_data = np.load('dataset_9_coordinates.npy')


with open('veri9.csv', 'w', newline='') as f:
    writer = csv.writer(f)


    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:

        writer.writerow(npy_data)

npy_data = np.load('dataset_10_coordinates.npy')


with open('veri10.csv', 'w', newline='') as f:
    writer = csv.writer(f)


    if np.ndim(npy_data) == 2:
        writer.writerows(npy_data)
    else:

        writer.writerow(npy_data)
