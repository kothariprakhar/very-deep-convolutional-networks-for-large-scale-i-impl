import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Check for device availability
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Model Architecture: VGG (General)
# ==========================================

# Configuration dictionary for different VGG variants.
# numbers: output channels for conv layer (3x3, stride 1, padding 1)
# 'M': MaxPool2d (2x2, stride 2)
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(VGG_CONFIGS[vgg_name])
        
        # Adaptive pooling allows the model to handle different input sizes 
        # while maintaining a fixed output size for the classifier.
        # For CIFAR-10 (32x32), standard pooling reduces size to 1x1 by the end.
        # For ImageNet (224x224), it reduces to 7x7.
        # Here we force a 1x1 output to make it compatible with 32x32 CIFAR images efficiently.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3 # RGB
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # VGG uses 3x3 kernels, stride 1, padding 1 to preserve spatial dims
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), # Modern addition: BatchNorm helps convergence significantly
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Kaiming / He Initialization (Standard for ReLU networks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ==========================================
# 2. Data Strategy: CIFAR-10
# ==========================================

def get_data_loaders(batch_size=128):
    print("Preparing Data...")
    # Standard normalization for CIFAR-10
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    # Downloads to './data' if not present
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

# ==========================================
# 3. Training Logic
# ==========================================

def train_model():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10 
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    
    # Load Data
    trainloader, testloader, classes = get_data_loaders(BATCH_SIZE)

    # Initialize VGG-16 (The most popular variant from the paper)
    model = VGG('VGG16', num_classes=10).to(device)
    
    # Loss and Optimizer (SGD with momentum as per paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=5e-4)
    
    # Learning rate scheduler (decay LR when loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print(f"\nStarting training for {EPOCHS} epochs on {device}...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i+1) % 100 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Step {i+1}/{len(trainloader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

        epoch_loss = running_loss / len(trainloader)
        scheduler.step(epoch_loss)
        print(f"End of Epoch {epoch+1} | Avg Loss: {epoch_loss:.4f} | Train Acc: {100.*correct/total:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.2f}s")
    
    # ==========================================
    # 4. Evaluation
    # ==========================================
    print("Evaluating on Test Set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    train_model()