import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 10
epoch_list = []
loss_list = []
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)

        output = autoencoder(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Вывод прогресса обучения
    print('Epoch [{}/{}], Loss: {:.4f}'.format(
        epoch + 1, num_epochs, loss.item()))
    epoch_list.append(epoch + 1)
    loss_list.append(loss.item())

# Визуализация прогресса обучения
plt.scatter(epoch_list, loss_list)
plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

data_iter = iter(test_loader)
images, labels = next(data_iter)

# Визуализация исходного изображения
plt.imshow(images[0].squeeze(), cmap="gray")
plt.axis('off')
plt.show()

with torch.no_grad():
    decoded = autoencoder.forward(images[0].view(images[0].size(0), -1))

decoded = decoded.numpy().reshape(decoded.size(0), 28, 28)
points_cloud = []
points_color = images[0].squeeze()
points_color1 = (np.vstack((images[0].flatten(), images[0].flatten(), images[0].flatten())).T + 1) / 2

for i in range(decoded.shape[0]):
    image = decoded[i]
    x, y = np.meshgrid(range(28), range(28))
    points = np.vstack((x.flatten(), y.flatten(), image.flatten())).T
    points_cloud.append(points)

points_cloud = np.concatenate(points_cloud, axis=0)

# Визуализация облака точек через matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_cloud[:, 0], points_cloud[:, 1], points_cloud[:, 2], c=points_color1, marker='.')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
