import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# model_name = 'attention_2'
#
# file_path = 'result/' + model_name
#
# attention_name1 = file_path + '_0.001_256_256.pkl'
# attention_loss1 = pickle.load(open(attention_name1, 'rb'))[0]
# attention_acc1 = np.mean(sorted(pickle.load(open(attention_name1, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc1 = format(attention_acc1, '.2f')
#
# attention_name2 = file_path + '_0.001_512_512.pkl'
# attention_loss2 = pickle.load(open(attention_name2, 'rb'))[0]
# attention_acc2 = np.mean(sorted(pickle.load(open(attention_name2, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc2 = format(attention_acc2, '.2f')
#
# attention_name3 = file_path + '_0.01_256_256.pkl'
# attention_loss3 = pickle.load(open(attention_name3, 'rb'))[0]
# attention_acc3 = np.mean(sorted(pickle.load(open(attention_name3, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc3 = format(attention_acc3, '.2f')
#
# attention_name4 = file_path + '_0.01_512_512.pkl'
# attention_loss4 = pickle.load(open(attention_name4, 'rb'))[0]
# attention_acc4 = np.mean(sorted(pickle.load(open(attention_name4, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc4 = format(attention_acc4, '.2f')
#
# X = range(1995)
#
# model_name = 'Attention_CN'
#
# plt.figure()
#
# plt.plot(X, sorted(attention_loss1, reverse=True)[5:], label='lr=0.001, emb=256, hid=256, acc=' + str(attention_acc1) + '%')
# plt.plot(X, sorted(attention_loss2, reverse=True)[5:], label='lr=0.001, emb=512, hid=512, acc=' + str(attention_acc2) + '%')
# plt.plot(X, sorted(attention_loss3, reverse=True)[5:], label='lr=0.01, emb=256, hid=256, acc=' + str(attention_acc3) + '%')
# plt.plot(X, sorted(attention_loss4, reverse=True)[5:], label='lr=0.01, emb=512, hid=512, acc=' + str(attention_acc4) + '%')
# plt.legend()
# plt.title(model_name + ' Loss & Test Accuracy')
#
# plt.show()














model_name = 'attetion'
file_path = 'result/finished/' + model_name

attention_name1 = file_path + '_0.01_256_256.pkl'
attention_loss1 = pickle.load(open(attention_name1, 'rb'))[0]
attention_acc1 = np.mean(sorted(pickle.load(open(attention_name1, 'rb'))[3], reverse=True)[:3]) * 100
attention_acc1 = format(attention_acc1, '.2f')


model_name = 'fusion_gru'
file_path = 'result/finished/' + model_name

attention_name2 = file_path + '_0.001_512_512.pkl'
attention_loss2 = pickle.load(open(attention_name2, 'rb'))[0]
attention_acc2 = np.mean(sorted(pickle.load(open(attention_name2, 'rb'))[3], reverse=True)[:3]) * 100
attention_acc2 = format(attention_acc2, '.2f')

model_name = 'naive_gru'
file_path = 'result/finished/' + model_name

attention_name3 = file_path + '_0.001_128_256.pkl'
attention_loss3 = pickle.load(open(attention_name3, 'rb'))[0]
attention_acc3 = np.mean(sorted(pickle.load(open(attention_name3, 'rb'))[3], reverse=True)[:3]) * 100
attention_acc3 = format(attention_acc3, '.2f')

# attention_name4 = file_path + '_0.01_512_512.pkl'
# attention_loss4 = pickle.load(open(attention_name4, 'rb'))[0]
# attention_acc4 = np.mean(sorted(pickle.load(open(attention_name4, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc4 = format(attention_acc4, '.2f')

X1 = range(995)
X2 = range(1995)

plt.figure()

plt.plot(X2, sorted(attention_loss1, reverse=True)[5:], label='Attention - LSTM, lr=0.01, e=h=256, acc=' + str(39.17) + '%')
plt.plot(X1, sorted(attention_loss3, reverse=True)[5:], label='Naive - GRU, lr=0.001, e=128, h=256, acc=' + str(37.97) + '%')
plt.plot(X1, sorted(attention_loss2, reverse=True)[5:], label='Fusion - GRU, lr=0.001, e=h=512, acc=' + str(35.70) + '%')
# plt.plot(X, sorted(attention_loss4, reverse=True)[5:], label='lr=0.01, emb=512, hid=512, acc=' + str(attention_acc4) + '%')
plt.legend()
plt.title('Training Loss & Averaged Test Accuracy')

plt.xlabel('Training Iteration')
plt.ylabel('Training Loss')

plt.show()






















# model_name = 'attetion'
# file_path = 'result/finished/' + model_name
#
# attention_name1 = file_path + '_0.01_256_256.pkl'
# attention_loss1 = pickle.load(open(attention_name1, 'rb'))[0]
# attention_acc1 = np.mean(sorted(pickle.load(open(attention_name1, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc1 = format(attention_acc1, '.2f')
#
#
# model_name = 'naive_gru'
# file_path = 'result/finished/' + model_name
#
# attention_name2 = file_path + '_0.001_128_256.pkl'
# attention_loss2 = pickle.load(open(attention_name2, 'rb'))[0]
# attention_acc2 = np.mean(sorted(pickle.load(open(attention_name2, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc2 = format(attention_acc2, '.2f')
#
# model_name = 'naive_rnn'
# file_path = 'result/finished/' + model_name
#
# attention_name3 = file_path + '_0.001_128_256.pkl'
# attention_loss3 = pickle.load(open(attention_name3, 'rb'))[0]
# attention_acc3 = np.mean(sorted(pickle.load(open(attention_name3, 'rb'))[3], reverse=True)[:3]) * 100
# attention_acc3 = format(attention_acc3, '.2f')
#
# # attention_name4 = file_path + '_0.01_512_512.pkl'
# # attention_loss4 = pickle.load(open(attention_name4, 'rb'))[0]
# # attention_acc4 = np.mean(sorted(pickle.load(open(attention_name4, 'rb'))[3], reverse=True)[:3]) * 100
# # attention_acc4 = format(attention_acc4, '.2f')
#
# X1 = range(995)
# X2 = range(1995)
#
# plt.figure()
#
# plt.plot(X2, sorted(attention_loss1, reverse=True)[5:], label='LSTM - Attention, lr=0.01, e=h=256, acc=' + str(38.85) + '%')
# plt.plot(X1, sorted(attention_loss3, reverse=True)[5:], label='GRU - Naive, lr=0.001, e=128, h=256, acc=' + str(38.79) + '%')
# plt.plot(X1, sorted(attention_loss2, reverse=True)[5:], label='RNN - Naive, lr=0.001, e=128, h=256, acc=' + str(33.17) + '%')
# # plt.plot(X, sorted(attention_loss4, reverse=True)[5:], label='lr=0.01, emb=512, hid=512, acc=' + str(attention_acc4) + '%')
# plt.legend()
# plt.title('Training Loss & Averaged Test Accuracy')
#
# plt.xlabel('Training Iteration')
# plt.ylabel('Training Loss')
#
# plt.show()

