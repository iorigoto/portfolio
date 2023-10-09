from dezero import Model
from dezero import SeqDataLoader
import numpy as np
import dezero.layers as L
import dezero.functions as F 
import dezero
import matplotlib.pyplot as plt

#-----------------------------------------------------------
#set hyper params
max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30 #length of BPTT

train_set = dezero.datasets.SinCurve(train=True)
#use dataloader for time series
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)

#-----------------------------------------------------------

class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y

#-----------------------------------------------------------

model = BetterRNN(hidden_size,1)
optimizer = dezero.optimizers.Adam().setup(model)

#start learning
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1
        
        #2:adjust the timing of Truncated BPTT
        #every 30th and last
        if count % bptt_length == 0 or count == seqlen:
            dezero.utils.plot_dot_graph(loss,to_file='LSTM.pdf') #plot a graph
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()#3:disconnect
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss)) 

#-----------------------------------------------------------
#noiseless cosin wave
#-----------------------------------------------------------

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()#model reset 
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1) #convert shape to (1,1)
        y = model(x)  
        pred_list.append(float(y.data))
plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
#-----------------------------------------------------------
