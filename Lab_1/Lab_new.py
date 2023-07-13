import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Cfg:
    lr = 0.05
    activate = 'Sigmoid'
    schedular = 'WarmupLinear'
    epoch = 50000

def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)
def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)
def show_result(x, y, pred_y, file_name = 'result.png'):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.savefig(file_name)
def show_curve(loss, file_name = 'loss', title = 'Learning Curve', x ='epoch', y = 'loss'):
    plt.figure()
    plt.title(title)
    plt.plot(loss)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(file_name)

def activate(x, activate = Cfg.activate):
    activation_functions = {
        'Sigmoid' : lambda x : 1.0/(1.0 + np.exp(-x)),
        'tanh' : lambda x : np.tanh(x),
        'ReLU': lambda x: np.maximum(0.0, x),
        'leaky_ReLU': lambda x: np.maximum(0.01 * x, x),
        'Identity' : lambda x : x
    }
    if activate in activation_functions:
        return activation_functions[activate](x)
    else:
        print('Activation function not found.')
def d_activate(x, activate = Cfg.activate):
    activation_functions = {
        'Sigmoid': lambda x: np.multiply(x, 1.0 - x),
        'tanh': lambda x: 1.0 - x ** 2,
        'ReLU': lambda x: np.where(x <= 0, 0.0, 1.0),
        'leaky_ReLU': lambda x: np.where(x <= 0.0, 0.01, 1.0),
        'Identity': lambda x: np.full_like(x, 1.0)
    }
    if activate in activation_functions:
        return activation_functions[activate](x)
    else:
        print('d_Activation function not found.')

class Layer:
    def __init__(self, input_number, output_number, activate = Cfg.activate):
        self.input_number = input_number
        self.output_number = output_number
        self.activate = activate
        self.generate_weight()
    def generate_weight(self):
        self.weight = np.random.normal(0, 1,(self.input_number + 1, self.output_number))
    def forward_once(self, input):
        self.input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
        self.output = np.matmul(self.input,self.weight)
        self.output = activate(self.output, self.activate)
        return self.output
    def backward_once(self,pre_loss_gradient):
        self.neuron_gradient = np.multiply(pre_loss_gradient,d_activate(self.output,self.activate))
        self.back_neuron_gradient = np.matmul(self.neuron_gradient, self.weight[:-1].T) 
        return self.back_neuron_gradient
    def update(self,lr):
        self.weight_gradient = np.matmul(self.input.T,self.neuron_gradient)
        self.weight = self.weight - lr * self.weight_gradient

class Network:
    def __init__(self, input_number, output_number, hidden_number, hidden_size,lr = Cfg.lr):
        self.input_number = input_number
        self.output_number = output_number
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.lr = lr
        self.generate_network()
    def generate_network(self):
        self.network = []
        self.network.append(Layer(self.input_number, self.hidden_size))
        for i in range(self.hidden_number - 1):
            self.network.append(Layer(self.hidden_size, self.hidden_size))  
        self.network.append(Layer(self.hidden_size, self.output_number, activate = 'Sigmoid'))
    def forward(self,input):
        for layer in self.network:
            input = layer.forward_once(input)
        return input
    def backward(self,gradient):
        for layer in reversed(self.network):
            gradient = layer.backward_once(gradient)
    def update(self):
        for layer in self.network:
            layer.update(self.lr)
    def schedular_step(self, step, total_step, schedular = 'None', warmup_step = 15000, gamma = 0.95):
        schedular_list = {
            'None': lambda lr: lr,
            'ExponentialLR': lambda lr: gamma * lr if step % 1000 == 0 else lr,
            'PolynomialLR': lambda lr: lr - gamma,
            'WarmupLinear': lambda lr: float(step) / float(max(1, warmup_step)) \
            if step < warmup_step \
            else max(0.0, float(total_step - step)) / float(max(1.0, total_step - warmup_step)),
        }
        if schedular in schedular_list:
            self.lr =  schedular_list[schedular](self.lr)
        else:
            print('Schedular function not found.')
    def loss_function(self, predict, label):
        return 1/2*np.mean((predict - label)**2)
    def d_loss(self, predict, label):
        return -(label - predict)/len(label.shape)
    
def predict2label(predict, threshold = 0.5):
    return np.where(predict > threshold, 1, 0)

lr = []
def train(model, data, epoch = Cfg.epoch):
    total_loss = []
    with tqdm(range(epoch)) as tqdm_loader:
        for i in tqdm_loader:
            lr.append(model.lr)
            predict = predict2label(model.forward(data["input"]))
            loss = model.loss_function(predict, data['label'])
            total_loss.append(loss)
            model.backward(model.d_loss(predict, data['label']))
            model.update()
            model.schedular_step(schedular = Cfg.schedular, step = i, total_step = epoch)
            accuracy = np.sum(predict == data['label']) / len(data['label'])
            tqdm_loader.set_postfix(epoch = i, loss = loss, lr = model.lr, accuracy = accuracy)
        return total_loss

def test(model, data):
    predict = predict2label(model.forward(data["input"]))
    return predict

data_linear = {}
data_linear['input'], data_linear['label'] = generate_linear()
model_linear = Network(input_number = 2, output_number = 1, hidden_number = 2, hidden_size = 4)
total_loss_linear = train(model_linear, data_linear)
predict = test(model_linear, data_linear)
show_result(data_linear['input'], data_linear['label'], predict, file_name = 'linear')
show_curve(total_loss_linear, file_name = 'linear_loss')


show_curve(lr, file_name = 'learning_rate', title = 'learning rate curve', x = 'epoch', y = 'lr')

data_XOR = {}
data_XOR['input'], data_XOR['label'] = generate_XOR_easy()
model_XOR = Network(input_number = 2, output_number = 1, hidden_number = 2, hidden_size = 4)
total_loss_XOR = train(model_XOR, data_XOR)
predict = test(model_XOR, data_XOR)
show_result(data_XOR['input'], data_XOR['label'], predict, file_name = 'XOR')
show_curve(total_loss_XOR, file_name = 'XOR_loss')