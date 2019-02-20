import numpy as np
#import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil


new_path = './track1/IMG/'
current_path = './out2rev/IMG/'


if not os.path.exists(new_path):
    os.makedirs(new_path)
    print('Folder created: ', new_path)


zero = 0
df = pd.read_csv('./out2rev/driving_log.csv', header=None)

counted_samples = np.unique(df.loc[:, 3], return_counts=True)

# zero = 0
# angles = []
# for row in train_samples:
#     #angles.append(np.round(float(row[3]), 20))
#     angles.append(float(row[3]))
#     if float(row[3]) == 0.0:
#         zero = zero + 1
#         if zero > 3000:
#             s = 0


print('Index:', len(counted_samples[0]))
max_ind = np.argmax(counted_samples[1])
print('Zero:', zero, 'Max index', max_ind,
      'Val max:', counted_samples[0][max_ind])

plt.figure(figsize=(20, 10))

index = np.arange(len(counted_samples[0]))
plt.bar(counted_samples[0], counted_samples[1],  color='b', width=0.005)
plt.ylabel('Counts', fontsize=15)
plt.title('Training examples')
plt.savefig('counted_samples.png')
plt.clf()

updated = pd.DataFrame()
total = 0

for i in range(0, len(counted_samples[0])):
    df_range = df[df[3] == counted_samples[0][i]]
    range_n = min(500, df_range.shape[0])
    total = total + range_n
    #print(counted_samples[0][i], ' added: ', range_n)
    updated = pd.concat([updated, df_range.sample(range_n)])


print('Total', total)
updated.to_csv('./track1/driving_log_updated.csv', index=False)


counted_bal = np.unique(updated.loc[:, 3], return_counts=True)
print('Index:', len(counted_bal[0]))
max_ind = np.argmax(counted_bal[1])
print('Zero:', zero, 'Max index', max_ind, ' Counts: ', counted_bal[1][max_ind],
      'Val max:', counted_bal[0][max_ind])


index = np.arange(len(counted_bal[0]))
plt.bar(counted_bal[0], counted_bal[1],  color='b', width=0.005)
plt.ylabel('Counts', fontsize=15)
plt.title('Updated training examples')
plt.savefig('counted_bal.png')

# ------------- Copying files
ind = 0


# for index, row in updated.iterrows():

#     file1 = row[0].split('/')[-1]
#     file2 = row[1].split('/')[-1]
#     file3 = row[2].split('/')[-1]

#     if os.path.isfile(current_path+file1):
#         shutil.move(current_path + file1, new_path + file1)
#         ind = ind + 1

#     else:
#         print('file not found')

#     if os.path.isfile(current_path+file2):
#         shutil.move(current_path + file2, new_path + file2)
#         ind = ind + 1

#     else:
#         print('file not found')

#     if os.path.isfile(current_path+file3):
#         shutil.move(current_path + file3, new_path + file3)
#         ind = ind + 1

#     else:
#         print('file not found')

#     # if ind > 20:
#      #   quit()


# print('Copied:', ind)
