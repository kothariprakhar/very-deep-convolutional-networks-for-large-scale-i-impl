import unittest
import torch
import torch.nn as nn
from contextlib import redirect_stdout
import io

# Assuming the user's code is strictly the class and function definitions provided.
# We import the VGG class and configs from the snippet.
# For the purpose of this test, we replicate the necessary class structure if it were in a separate file,
# or we assume this test file is appended to the source.
# Below, we treat the provided code as imported or available in scope.

# --- REPLICATING MINIMAL DEPENDENCIES FOR INDEPENDENT EXECUTION IF NEEDED ---
# In a real scenario, we would `from model_file import VGG`
# Here we define the content to ensure the test runs in this environment.

VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(VGG_CONFIGS[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
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
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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

class TestVGGImplementation(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 10
        self.batch_size = 4

    def test_vgg11_instantiation(self):
        """Test if VGG11 instantiates correctly without errors."""
        model = VGG('VGG11', num_classes=self.num_classes)
        self.assertIsInstance(model, VGG)
        # VGG11 has 8 conv layers + 3 FC layers = 11 weight layers.
        # In features: 8 convs.
        conv_count = sum(1 for m in model.features.modules() if isinstance(m, nn.Conv2d))
        self.assertEqual(conv_count, 8, "VGG11 should have 8 convolutional layers")

    def test_vgg16_instantiation(self):
        """Test if VGG16 instantiates correctly."""
        model = VGG('VGG16', num_classes=self.num_classes)
        # VGG16 has 13 conv layers.
        conv_count = sum(1 for m in model.features.modules() if isinstance(m, nn.Conv2d))
        self.assertEqual(conv_count, 13, "VGG16 should have 13 convolutional layers")

    def test_forward_pass_cifar_shape(self):
        """Test forward pass with standard CIFAR-10 shape (32x32)."""
        model = VGG('VGG16', num_classes=self.num_classes).to(self.device)
        dummy_input = torch.randn(self.batch_size, 3, 32, 32).to(self.device)
        
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_forward_pass_imagenet_shape(self):
        """Test forward pass with larger shape (224x224). 
        Ensures AdaptiveAvgPool handles inputs larger than 32x32 without crashing."""
        model = VGG('VGG16', num_classes=self.num_classes).to(self.device)
        # 224x224 input
        dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
        
        try:
            output = model(dummy_input)
        except Exception as e:
            self.fail(f"Model failed on large input 224x224: {e}")
            
        self.assertEqual(output.shape, (2, self.num_classes))

    def test_gradient_flow(self):
        """Ensure gradients are propagated backwards."""
        model = VGG('VGG11', num_classes=self.num_classes).to(self.device)
        dummy_input = torch.randn(2, 3, 32, 32).to(self.device)
        dummy_target = torch.randint(0, self.num_classes, (2,)).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

        # Check if the first conv layer weights have a gradient
        first_conv = next(m for m in model.features.modules() if isinstance(m, nn.Conv2d))
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertNotEqual(torch.sum(first_conv.weight.grad), 0)

    def test_initialization(self):
        """Verify that weights are not all zeros or NaNs."""
        model = VGG('VGG11', num_classes=self.num_classes)
        for param in model.parameters():
            self.assertFalse(torch.isnan(param).any())
            # Biases might be zero, but weights shouldn't be all zero usually
            if len(param.shape) > 1: # Weight matrix/tensor
                self.assertNotEqual(torch.sum(torch.abs(param)), 0)

if __name__ == '__main__':
    unittest.main()
