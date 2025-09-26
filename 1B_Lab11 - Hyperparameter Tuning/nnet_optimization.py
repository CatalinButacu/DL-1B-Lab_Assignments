import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import optuna

class NNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super().__init__()
        layers = [nn.Flatten()]
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def train_model(self, train_dataset, optimizer_name, learn_rate, batch_size):

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

        loss_func = nn.CrossEntropyLoss()
        optimizer = getattr(optim, optimizer_name)(self.parameters(), lr = learn_rate)

        self.train()
        for epoch in range(5):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

    def eval_model(self, test_dataset, batch_size = 64):
        test_loader = DataLoader(test_dataset, batch_size = batch_size)
        self.eval()
        total = 0
        wrong = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.forward(images)
                predicted = torch.argmax(outputs, dim = 1)
                wrong += (predicted != labels).sum().item()
                total += labels.shape[0]
        error = wrong / total
        return error

class CNNModel(nn.Module):
    def __init__(self, input_shape, conv_params_list, fc_hidden_sizes, dropout_rate):
        super().__init__()
        self.input_shape = input_shape 
        
        conv_layers_list = []
        current_channels = input_shape[0]
        current_h, current_w = input_shape[1], input_shape[2]

        for params in conv_params_list:
            conv_layers_list.append(nn.Conv2d(current_channels, params['out_channels'], 
                                              kernel_size=params['kernel_size'], 
                                              padding=(params['kernel_size']-1)//2))
            conv_layers_list.append(nn.ReLU())
            current_channels = params['out_channels']

            if params['pool_type'] != 'none':
                if params['pool_type'] == 'max':
                    conv_layers_list.append(nn.MaxPool2d(kernel_size=params['pool_kernel_size'], 
                                                         stride=params['pool_kernel_size']))
                elif params['pool_type'] == 'avg':
                    conv_layers_list.append(nn.AvgPool2d(kernel_size=params['pool_kernel_size'], 
                                                         stride=params['pool_kernel_size']))
                current_h //= params['pool_kernel_size']
                current_w //= params['pool_kernel_size']
        
        self.conv_layers = nn.Sequential(*conv_layers_list)
        
        conv_output_size = current_channels * current_h * current_w

        fc_layers_list = [nn.Flatten()]
        current_size = conv_output_size
        for hidden_size in fc_hidden_sizes:
            fc_layers_list.append(nn.Linear(current_size, hidden_size))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        fc_layers_list.append(nn.Linear(current_size, 10))
        
        self.fc_layers = nn.Sequential(*fc_layers_list)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def train_model(self, train_dataset, optimizer_name, learn_rate, batch_size, num_epochs=5):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        optimizer = getattr(optim, optimizer_name)(self.parameters(), lr=learn_rate)

        self.train()
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

    def eval_model(self, test_dataset, batch_size=64):
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        self.eval()
        total = 0
        wrong = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.forward(images)
                predicted = torch.argmax(outputs, dim=1)
                wrong += (predicted != labels).sum().item()
                total += labels.shape[0]
        error = wrong / total
        return error

class Objective:
    def __init__(self, train_dataset, validation_dataset):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def __call__(self, trial):
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3) 
        conv_params_list = []
        current_h, current_w = 28, 28 # MNIST dimensions (Height, Width)

        for i in range(n_conv_layers):
            out_channels = trial.suggest_categorical(f'conv_l{i}_out_channels', [16, 32, 64])
            kernel_size = trial.suggest_categorical(f'conv_l{i}_kernel_size', [3, 5])
            pool_type = 'none'
            pool_kernel_size = 1

            if current_h >= 2 and current_w >= 2:
                pool_type_candidate = trial.suggest_categorical(f'conv_l{i}_pool_type', ['max', 'avg', 'none'])
                if pool_type_candidate != 'none':
                    possible_pool_kernels = []
                    if current_h >= 2 and current_w >= 2:
                        possible_pool_kernels.append(2)
                    if current_h >= 3 and current_w >= 3:
                        possible_pool_kernels.append(3)
                    
                    if possible_pool_kernels:
                        pool_kernel_size_candidate = trial.suggest_categorical(f'conv_l{i}_pool_kernel_size', possible_pool_kernels)
                        
                        if current_h // pool_kernel_size_candidate > 0 and current_w // pool_kernel_size_candidate > 0:
                            pool_type = pool_type_candidate
                            pool_kernel_size = pool_kernel_size_candidate
                            
            
            conv_params_list.append({
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'pool_type': pool_type,
                'pool_kernel_size': pool_kernel_size
            })
            
            if pool_type != 'none':
                current_h //= pool_kernel_size
                current_w //= pool_kernel_size
            
            if current_h == 0 or current_w == 0: # Stop adding conv layers if dimensions are too small
                break
        
        
        n_fc_layers = trial.suggest_int('n_fc_layers', 1, 2)
        fc_hidden_sizes_list = []
        
        if current_h > 0 and current_w > 0:
            for i in range(n_fc_layers):
                fc_hidden_sizes_list.append(trial.suggest_int(f'fc_l{i}_hidden_size', 32, 128))
        else:
            if not fc_hidden_sizes_list:
                 fc_hidden_sizes_list.append(10)

        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learn_rate = trial.suggest_float('learn_rate', 1e-4, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        num_epochs_train = trial.suggest_int('num_epochs', 3, 7)

        if current_h == 0 or current_w == 0:
            return float('inf')
            
        model = CNNModel(input_shape=(1, 28, 28), 
                         conv_params_list=conv_params_list,
                         fc_hidden_sizes=fc_hidden_sizes_list,
                         dropout_rate=dropout_rate)
        
        model.train_model(self.train_dataset, optimizer_name, learn_rate, batch_size, num_epochs=num_epochs_train)
        err = model.eval_model(self.validation_dataset, batch_size)
        
        return err

if __name__ == '__main__':

    transform = transforms.ToTensor()
    full_dataset = torchvision.datasets.MNIST(root = './data', 
                                               train = True, download = True,
                                               transform = transform)
    test_dataset = torchvision.datasets.MNIST('./data',  
                                          train = False, download = True, 
                                          transform = transform)

    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    study = optuna.create_study(study_name = 'CNN Optimizer', direction='minimize')
    study.optimize(Objective(train_dataset, validation_dataset), n_trials = 10)

    print('Best trial for CNN:')
    trial = study.best_trial
    print(f'   Value: {trial.value:.4f}')
    print(f'   Params:')
    for key, value in trial.params.items():
        print(f'      {key} : {value}')




    