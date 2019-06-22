import numpy as np
import matplotlib.pyplot as plt
import pickle


# name = 'attention_2_0.001_512_512'
#
# with open('result/' + name + '.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# train_loss = data[0]
# train_acc = data[1]
# test_loss = data[2]
# test_acc = data[3]
#
# plt.subplot(2, 2, 1)
# plt.plot(range(len(train_loss)), train_loss, 'ro')
# plt.title('train_loss')
#
#
# plt.subplot(2, 2, 2)
# plt.plot(range(len(train_acc)), train_acc, 'ro')
# plt.title('train_acc')
#
#
# plt.subplot(2, 2, 3)
# plt.plot(range(len(test_loss)), test_loss, 'ro')
# plt.title('test_loss')
#
#
# plt.subplot(2, 2, 4)
# plt.plot(range(len(test_acc)), test_acc, 'ro')
# plt.title('test_acc ' + str(np.mean(sorted(test_acc, reverse=True)[:3])))
#
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.show()

# lstm
l1 = [
39.00,
40.67,
41.00,
40.00,
36.00,
37.67,
39.33,
39.67,
37.00,
39.33,
39.33,
38.67,
37.33,
39.67,
38.00,
39.00
]

# gru
l2 = [
37.00,
40.67,
39.00,
37.67,
41.67,
40.33,
36.67,
37.33
]

# rnn
l3 = [
35.67,
33.33,
36.67,
14.00,
39.67,
39.00,
35.33,
31.67
]

print(np.mean(l1))
print(np.mean(l2))
print(np.mean(l3))

