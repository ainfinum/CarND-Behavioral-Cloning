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
    df_temp = df[df[3] == counted_samples[0][i]]
    min_set = min(500, df_temp.shape[0])
    total = total + min_set
    updated = pd.concat([updated, df_temp.sample(min_set)])


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
