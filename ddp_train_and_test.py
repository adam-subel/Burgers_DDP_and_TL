import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.sparse import linalg
import scipy.io as sio
import math
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import rmsprop, SGD, Adagrad, Adadelta
from scipy.io import savemat
from scipy.io import loadmat
from scipy.fftpack import fft, ifft

def swish(x):
   beta = 1.0
   return beta * x * keras.backend.sigmoid(x)


train_num = 500000
region = "13"

train_region = 1000000
train_start = 0
num_pred = 20000

def normalize_data(data):

  std_data = np.std(data)
  mean_data = np.mean(data)

  norm_data = (data-mean_data)/std_data

  return norm_data, mean_data, std_data

def shift_data(data1,data2):
  shifts = np.random.randint(0,data1.shape[1],data1.shape[0])
  for i in range(data1.shape[0]):
    data1[i,:] = np.concatenate((data1[i,shifts[i]:], data1[i,:shifts[i]]))
    data2[i,:] = np.concatenate((data2[i,shifts[i]:], data2[i,:shifts[i]]))

  return data1, data2

u_bar_dict = sio.loadmat("./u_bar_region_"+ region +".mat")

full_input=u_bar_store=u_bar_dict['u_bar'].transpose()



full_output = sio.loadmat("./PI_region_" + region +".mat")
full_output=full_output['PI'].transpose()

full_input[:train_region,:], full_output[:train_region,:] = shift_data(full_input[:train_region,:],
                                                                 full_output[:train_region,:])



norm_input, mean_input, std_input = normalize_data(full_input[:train_region,:])

norm_output, mean_output, std_output = normalize_data(full_output[:train_region,:])


training_input = norm_input
training_output = norm_output

print('shape of input')
print(np.shape(training_input))


print('shape of output')
print(np.shape(training_output))


index=np.random.permutation(train_region)


print(std_input)
print(std_output)

print(mean_input)
print(mean_output)


input_train=training_input[index[0:train_num],:]
output_train=training_output[index[0:train_num],:]

test_input=training_input[index[train_num:(train_num+num_pred)],:]
test_output=training_output[index[train_num:(train_num+num_pred)],:]


model = Sequential()

model.add(Dense(128,input_shape=(128,),activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(250,activation=swish))
model.add(Dense(128,activation=None))

model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(input_train, output_train,nb_epoch=100,batch_size=200,shuffle=True,validation_split=0.2)

model.save_weights('./weights_trained_ANN')

pred_start = train_region + 50000

s=20

NX = 128
nu = 2e-2

dt = s*1e-2

Lx    = 100
dx    = Lx/NX
x     = np.linspace(0, Lx-dx, num=NX)
kx    = (2*math.pi/Lx)*np.concatenate((np.arange(0,NX/2+1,dtype=float),np.arange((-NX/2+1),0,dtype=float))).reshape([NX,1])


maxit=100000

D1 = 1j*kx
D2 = kx*kx
D1 = D1.reshape([NX,1])
D2 = D2.reshape([NX,1])
D2_tensor = np.float32((D2[0:int(NX/2)]-np.mean(D2[0:int(NX/2)])/np.std(D2[0:int(NX/2)])))

D2x = 1 + 0.5*dt*nu*D2


u_store = np.zeros((NX,maxit))
sub_store = np.zeros((NX,maxit))

reg = 13

force_dict = sio.loadmat("./f_bar_all_regions.mat")
force_bar=force_dict['f_bar'][:,int((reg-1)*12500)+int(pred_start/s):]

u_old = full_input[pred_start-1,:].reshape([NX,1])
u = full_input[pred_start,:].reshape([NX,1])
u_fft = fft(u,axis=0)
u_old_fft = fft(u_old,axis=0)
subgrid_prev_n = model.predict(((u_old-mean_input)/std_input).reshape((1,128))).reshape(128,1)
subgrid_prev_n = subgrid_prev_n*std_output+mean_output

for i in range(maxit):
  subgrid_n = model.predict(((u-mean_input)/std_input).reshape((1,128))).reshape(128,1)
  subgrid_n = subgrid_n*std_output+mean_output

  force=force_bar[:,i].reshape((NX,1))

  F = D1*fft(.5*(u**2),axis=0)
  F0 = D1*fft(.5*(u_old**2),axis=0)

  uRHS = -0.5*dt*(3*F- F0) - 0.5*dt*nu*(D2*u_fft)  + u_fft + dt*fft(force,axis=0) \
             -fft(dt*3/2*subgrid_n + 1/2*dt*subgrid_prev_n,axis = 0)


  subgrid_prev_n = subgrid_n
  u_old_fft = u_fft
  u_old = u

  u_fft = uRHS/D2x.reshape([NX,1])
  u = np.real(ifft(u_fft,axis=0))
  u_store[:,i] = u.squeeze()
  sub_store[:,i] = subgrid_n.squeeze()

sio.savemat('./DDP_results_trained_'+str(int(train_num/1000))+'_region_' + region + '_new.mat',
           {'u_pred':u_store, 'sub_pred':sub_store})
