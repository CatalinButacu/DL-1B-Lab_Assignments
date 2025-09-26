import torch
import torch.nn as nn
import torch.optim as optim

class DynamicCNN(nn.Module):
    def __init__(self, num_classes=3, image_size=64, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.current_stage = 0 # 0: initial, 1: easy, 2: medium, 3: hard

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()        
        self.relu = nn.ReLU()
        self.dropout = dropout

        self._build_initial_model() 

    def _get_flat_features(self, current_image_size, out_channels_last_conv):
        # Calculate the size of the flattened features after conv and pool layers        
        
        if self.current_stage == 1:
            # Stage 1: conv1 -> pool1
            return out_channels_last_conv * (current_image_size // 2) * (current_image_size // 2)
        
        elif self.current_stage == 2:
            # Stage 2: conv1 -> pool1 -> conv2 -> pool2
            return out_channels_last_conv * (current_image_size // 4) * (current_image_size // 4)
        
        elif self.current_stage == 3:
            # Stage 3: conv1 -> pool1 -> conv2 -> pool2 -> conv3 (no pool3 in this example)
            return out_channels_last_conv * (current_image_size // 4) * (current_image_size // 4)

        else:
            print("Error: Invalid stage")
            return 0 

    def _build_initial_model(self):
        self.current_stage = 1        
        # Conv1, Pool1, FC
        self.conv_layers.append(nn.Conv2d(1, 16, kernel_size=5, padding='same'))
        self.conv_layers.append(nn.BatchNorm2d(16))
        self.conv_layers.append(nn.MaxPool2d(2))
        flat_features = self._get_flat_features(self.image_size, 16)
        self.fc_layers.append(nn.Linear(flat_features, self.num_classes))

    def _increase_complexity_stage2(self):
        self.current_stage = 2
        # Remove old FC layer
        old_fc = self.fc_layers.pop(0) 
        del old_fc

        # Conv2, Pool2, FC
        self.conv_layers.append(nn.Conv2d(16, 32, kernel_size=3, padding='same'))
        self.conv_layers.append(nn.BatchNorm2d(32))
        self.conv_layers.append(nn.MaxPool2d(2))        
        flat_features = self._get_flat_features(self.image_size, 32)
        self.fc_layers.append(nn.Linear(flat_features, self.num_classes))
        self.dropout = nn.Dropout(0.2)

    def _increase_complexity_stage3(self):
        self.current_stage = 3
        # Remove old FC layer
        old_fc = self.fc_layers.pop(0)
        del old_fc

        # Conv3, FC
        self.conv_layers.append(nn.Conv2d(32, 64, kernel_size=3, padding='same')) 
        self.conv_layers.append(nn.BatchNorm2d(64))
        flat_features = self._get_flat_features(self.image_size, 64) 
        self.fc_layers.append(nn.Linear(flat_features, self.num_classes))

    def forward(self, x):
        conv_idx = 0
        if self.current_stage >= 1: # Stage 1 layers
            x = self.relu(self.conv_layers[conv_idx+1](self.conv_layers[conv_idx](x)))
            x = self.conv_layers[conv_idx+2](x) # pool1
            conv_idx += 3
        
        if self.current_stage >= 2: # Stage 2 additional layers
            x = self.relu(self.conv_layers[conv_idx+1](self.conv_layers[conv_idx](x)))
            x = self.conv_layers[conv_idx+2](x) # pool2
            conv_idx += 3

        if self.current_stage >= 3: # Stage 3 additional layers
            x = self.relu(self.conv_layers[conv_idx+1](self.conv_layers[conv_idx](x))) 

        if self.dropout:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten for FC
        x = self.fc_layers[0](x)
        return x

    def train_model(self, easy_loader, medium_loader, hard_loader, num_epochs_per_stage=10, learn_rate=0.001):
        loss_fn = nn.CrossEntropyLoss()    
        history = {'overall_epochs': [], 'overall_loss': [], 'overall_train_accuracy': []}
        overall_epoch_count = 0

        stages = [
            {'loader': easy_loader, 'name': 'Easy Data', 'stage_num': 1, 'complexity_fn': None},
            {'loader': medium_loader, 'name': 'Medium Data', 'stage_num': 2, 'complexity_fn': self._increase_complexity_stage2},
            {'loader': hard_loader, 'name': 'Hard Data', 'stage_num': 3, 'complexity_fn': self._increase_complexity_stage3}
        ]
        
        # Loop through each stage
        for stage in stages:
            # Increase model complexity if needed (skip for first stage)
            if stage['complexity_fn']:
                stage['complexity_fn']()
            
            # Initialize optimizer for this stage
            optimizer = optim.Adam(self.parameters(), lr=learn_rate)
            print(f"\n--- Training Stage {stage['stage_num']} ({stage['name']}) ---")
            
            # Train for specified number of epochs
            for epoch in range(num_epochs_per_stage):
                self.train()
                total_loss_stage = 0
                correct_train_stage = 0
                total_train_stage = 0
                
                # Training loop
                for inputs, labels in stage['loader']:
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss_stage += loss.item()
                    _, predicted_train = torch.max(outputs.data, 1)
                    total_train_stage += labels.size(0)
                    correct_train_stage += (predicted_train == labels).sum().item()
                
                # Calculate and record metrics
                overall_epoch_count += 1
                avg_loss_stage = total_loss_stage / len(stage['loader'])
                epoch_train_accuracy_stage = 100 * correct_train_stage / total_train_stage
                history['overall_epochs'].append(overall_epoch_count)
                history['overall_loss'].append(avg_loss_stage)
                history['overall_train_accuracy'].append(epoch_train_accuracy_stage)
                
                # Print progress
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Stage {stage['stage_num']} Ep {epoch + 1}/{num_epochs_per_stage}, Loss: {avg_loss_stage:.4f}, Acc: {epoch_train_accuracy_stage:.2f}%")
        
        return history

    def evaluate_model(self, loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy




        







