import os

i = 0

# train_file = open("meta_train.csv", "w")
# test_file = open("meta_test.csv", "w")
# val_file = open("meta_val.csv", "w")
#
# train_file2 = open("meta_train2.csv", "w")
# test_file2 = open("meta_test2.csv", "w")
# val_file2 = open("meta_val2.csv", "w")

# gen_file = open("meta_gen.csv", "w")

vox_train_file = open("meta_vox_train.csv", 'w')
vox_val_file = open("meta_vox_val.csv", 'w')


tracks = os.listdir("E:/AEar/datasets/dataset_25_full_VOXOnly/tracks")
labels = os.listdir("E:/AEar/datasets/dataset_25_full_VOXOnly/masks")

for track in tracks:
    if i < 54000:
        vox_train_file.write(
            "E:/AEar/datasets/dataset_25_full_VOXOnly/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_VOXOnly/masks/" +
            labels[i] + '\n')
        i += 1
    else:
        vox_val_file.write(
            "E:/AEar/datasets/dataset_25_full_VOXOnly/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_VOXOnly/masks/" +
            labels[i] + '\n')
        i += 1

# for track in tracks:
#     if i < 60000:
#         train_file.write("E:/AEar/datasets/dataset_25_full_FIX/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_FIX/masks/" + labels[i] + '\n')
#         i += 1
#     elif i < 65000:
#         test_file.write("E:/AEar/datasets/dataset_25_full_FIX/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_FIX/masks/" + labels[i] + '\n')
#         i += 1
#     elif i < 70000:
#         val_file.write("E:/AEar/datasets/dataset_25_full_FIX/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_FIX/masks/" + labels[i] + '\n')
#         i += 1
#     elif i < 75000:
#         val_file2.write(
#             "E:/AEar/datasets/dataset_25_full_FIX/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_FIX/masks/" +
#             labels[i] + '\n')
#         i += 1
#     elif i < 80000:
#         test_file2.write(
#             "E:/AEar/datasets/dataset_25_full_FIX/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_FIX/masks/" +
#             labels[i] + '\n')
#         i += 1
#     else:
#         train_file2.write(
#             "E:/AEar/datasets/dataset_25_full_FIX/tracks/" + track + "," + "E:/AEar/datasets/dataset_25_full_FIX/masks/" +
#             labels[i] + '\n')
#         i += 1
#
