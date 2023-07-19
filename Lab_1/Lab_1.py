import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math

class Cfg:
    lr = 0.1
    epoch = 500
    warmup_step = 150
    seed = '312551116'
    activate = 'Sigmoid'
    schedular = 'ExponentialLR'
    loss_function = 'MSE'
    epislon = 1e-5
    optimizer = 'Adam'

def setSeed(seed = Cfg.seed):
    seed = math.prod([ord(i) for i in seed])%(2**32)
    random.seed(seed)
    np.random.seed(seed)

setSeed(seed = Cfg.seed)

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

data_linear = {}
data_linear['input'], data_linear['label'] = generate_linear(n = 100)
data_XOR = {}
data_XOR['input'], data_XOR['label'] = generate_XOR_easy()

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
def show_curve(loss, file_name = 'loss', title = 'Loss Curve', x = 'epoch', y = 'loss'):
    plt.figure()
    plt.title(title)
    for i in loss:
        plt.plot(loss[i], label = i)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.savefig(file_name)

def activate(x, activate = Cfg.activate):
    activation_functions = {
        'Sigmoid' : lambda x : 1.0/(1.0 + np.exp(-x)),
        'tanh' : lambda x : np.tanh(x),
        'ReLU': lambda x: np.maximum(0.0, x),
        'leaky ReLU': lambda x: np.maximum(0.01 * x, x),
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
        'leaky ReLU': lambda x: np.where(x <= 0.0, 0.01, 1.0),
        'Identity': lambda x: np.full_like(x, 1.0)
    }
    if activate in activation_functions:
        return activation_functions[activate](x)
    else:
        print('d_Activation function not found.')

def cal_loss(predict, label, loss_function, epislon = Cfg.epislon):
    loss_functions = {
        'MSE': lambda label, predict: np.mean(1/2*(predict - label)**2),
        'Binary cross entropy': lambda label, predict: \
        -np.mean(label * np.log(predict + epislon) + (1 - label) * np.log(1 - predict + epislon))
    }   
    if loss_function in loss_functions:
        return loss_functions[loss_function](label, predict)
    else:
        print('Loss function not found.')
def d_loss(predict, label, loss_function, epislon = Cfg.epislon):
    loss_functions = {
        'MSE': lambda label, predict : -(label - predict)/len(label.shape),
        'Binary cross entropy': lambda label, predict: -(label / (predict + epislon) - ((1 - label) / (1 - predict + epislon)))
    }
    if loss_function in loss_functions:
        return loss_functions[loss_function](label, predict)
    else:
        print('d_Loss function not found.')

class Layer:
    def __init__(self, input_number, output_number, activate):
        self.input_number = input_number
        self.output_number = output_number
        self.activate = activate
        self.v = np.zeros((input_number + 1, output_number))
        self.m = np.zeros((input_number + 1, output_number))
        self.generate_weight()
    def generate_weight(self):
        self.weight = np.random.normal(0, 1,(self.input_number + 1, self.output_number))
    def forward_once(self, input):
        self.input = np.append(input, np.ones((input.shape[0], 1)), axis = 1)
        self.output = np.matmul(self.input,self.weight)
        self.output = activate(self.output, self.activate)
        return self.output
    def backward_once(self,pre_gradient):
        self.current_gradient = np.multiply(pre_gradient,d_activate(self.output,self.activate))
        self.next_gradient = np.matmul(self.current_gradient, self.weight[:-1].T) 
        return self.next_gradient
    def update_once(self,lr, optimizer):
        self.weight_gradient = np.matmul(self.input.T, self.current_gradient)
        self.weight = self.weight - self.optimizer(lr = lr, optimizer = optimizer)
    def optimizer(self, lr, optimizer, beta = 0.9):
        if optimizer == 'gradient descent':
            return lr * self.weight_gradient
        elif optimizer == 'momentum':
            self.v = beta * self.v - lr * self.weight_gradient
            return -self.v
        elif optimizer == 'adagrad':
            self.v  = self.v + self.weight_gradient**2
            return (lr / np.sqrt(self.v) + Cfg.epislon) * self.weight_gradient
        elif optimizer == 'Adam':
            self.m = beta * self.m + (1 - beta) * self.weight_gradient
            self.v = beta * self.v + (1 - beta) * (self.weight_gradient ** 2)
            m_hat = self.m / (1 - beta)
            v_hat = self.v / (1 - beta)
            return (lr * m_hat  / np.sqrt(v_hat + Cfg.epislon))
class Network:
    def __init__(self, input_number, output_number, hidden_number, hidden_size, lr = Cfg.lr, optimizer = Cfg.optimizer, activate = Cfg.activate):
        self.input_number = input_number
        self.output_number = output_number
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.lr = lr
        self.activate = activate
        self.generate_network()
    def generate_network(self):
        self.network = []
        self.network.append(Layer(self.input_number, self.hidden_size, activate = self.activate))
        if self.hidden_number > 0:
            for i in range(self.hidden_number - 1):
                self.network.append(Layer(self.hidden_size, self.hidden_size, activate = self.activate))  
        self.network.append(Layer(self.hidden_size, self.output_number, activate = 'Sigmoid'))
    def forward(self,input):
        for layer in self.network:
            input = layer.forward_once(input)
        return input
    def backward(self,gradient):
        for layer in reversed(self.network):
            gradient = layer.backward_once(gradient)
    def update(self, optimizer = Cfg.optimizer):
        for layer in self.network:
            layer.update_once(self.lr, optimizer)
    def schedular_step(self, step, total_step, warmup_step, schedular = Cfg.schedular, gamma_ex = 0.95, gamma_lr = 0.01):
        schedular_list = {
            'None': lambda lr: lr,
            'ExponentialLR': lambda lr: gamma_ex * lr if step % (Cfg.epoch/50) == 0 else lr,
            'PolynomialLR': lambda lr: max(lr - gamma_lr, 0.001) if step % (Cfg.epoch/50) == 0 else lr,
            'WarmupLinear': lambda lr: float(step) / float(max(1, warmup_step)) \
            if step < warmup_step \
            else max(0.0, float(total_step - step)) / float(max(1.0, total_step - warmup_step)),
        }
        if schedular in schedular_list:
            self.lr =  schedular_list[schedular](self.lr)
        else:
            print('Schedular function not found.')
  
def predict2label(predict, threshold = 0.5):
    return np.where(predict > threshold, 1, 0)

def train(model, data, warmup_step = Cfg.epoch/2, schedular = Cfg.schedular, epoch = Cfg.epoch, loss_function = Cfg.loss_function):
    total_loss = []
    with tqdm(range(epoch)) as tqdm_loader:
        for i in tqdm_loader:
            predict = model.forward(data["input"])
            loss = cal_loss(loss_function = loss_function, predict = predict, label = data['label'])
            model.backward(d_loss(loss_function = loss_function, predict = predict, label = data['label']))
            model.update()
            model.schedular_step(schedular = schedular, step = i, total_step = epoch, warmup_step = Cfg.warmup_step)
            predict_label = predict2label(predict)
            accuracy = np.sum(predict_label == data['label']) / len(data['label'])
            tqdm_loader.set_postfix(epoch = i, loss = loss, lr = model.lr, accuracy = accuracy)
            total_loss.append(loss)
        return total_loss

def test(model, data):
    predict = model.forward(data["input"])
    predict_label = predict2label(predict)
    return predict, predict_label

total_loss = {}

model_linear = Network(input_number = 2, output_number = 1, hidden_number = 2, hidden_size = 2, activate = 'Sigmoid')
total_loss['Linear'] = train(model_linear, data_linear, epoch = Cfg.epoch)
predict, predict_label = test(model_linear, data_linear)
accuracy = np.sum(predict_label == data_linear['label']) / len(data_linear['label'])
loss = cal_loss(loss_function = Cfg.loss_function, predict = predict, label = data_linear['label'])
show_result(data_linear['input'], data_linear['label'], predict_label, file_name = 'Linear')
for i in range(predict.shape[0]):
    print('Iter {:2}  | Ground truth: {} | prediction: {:.15f}'.format(i, data_linear['label'].item(i), predict.item(i)))
print('loss = {:.15f} accuracy = {}'.format(loss, accuracy))

model_XOR = Network(input_number = 2, output_number = 1, hidden_number = 2, hidden_size = 4, activate = 'Sigmoid')
total_loss['XOR'] = train(model_XOR, data_XOR, epoch = Cfg.epoch)
predict, predict_label = test(model_XOR, data_XOR)
accuracy = np.sum(predict_label == data_XOR['label']) / len(data_XOR['label'])
loss = cal_loss(loss_function = Cfg.loss_function, predict = predict, label = data_XOR['label'])
show_result(data_XOR['input'], data_XOR['label'], predict_label, file_name = 'XOR')
for i in range(predict.shape[0]):
    print('Iter {:2}  | Ground truth: {} | prediction: {:.15f}'.format(i, data_linear['label'].item(i), predict.item(i)))
print('loss = {:.15f} accuracy = {}'.format(loss, accuracy))

show_curve(loss = total_loss, file_name = 'loss', title = 'Loss curve')