import ximu3
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ===============================
# 1D CNN Model Definition
# ===============================
class Simple1DCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1209, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CircularBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def append(self, item):
        self.buffer.append(item)  # Automatically removes oldest when full
    
    def get_all(self):
        return np.array(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

class CounterTrigger:
   def __init__(self, counter=0, trigger=False):
       self.counter = counter
       self.trigger = trigger
   
   def increment(self):
       """Increment the counter by 1"""
       self.counter += 1
   
   def decrement(self):
       """Decrement the counter by 1"""
       self.counter -= 1
   
   def set_trigger(self, value):
       """Set the trigger to True or False"""
       self.trigger = value
   
   def toggle_trigger(self):
       """Toggle the trigger between True and False"""
       self.trigger = not self.trigger
   
   def reset(self):
       """Reset counter to 0 and trigger to False"""
       self.counter = 0
       self.trigger = False
   
   def __str__(self):
       """String representation for easy printing"""
       return f"Counter: {self.counter}, Trigger: {self.trigger}"

THRESHOLD = 2
ATTACK = 1612
DECAY = 3225

ct = CounterTrigger()
arr = np.array([])
attack_buffer = CircularBuffer(ATTACK)
decay_buffer = CircularBuffer(DECAY)

surfaces = ["soft", "medium", "hard", "airtap", "tapup", "air"]

def inertial_callback(message):
    accelerometer = np.array([message.accelerometer_x, message.accelerometer_y, message.accelerometer_z])
    magnitude = np.linalg.norm(accelerometer)

    global arr

    if not ct.trigger and magnitude > THRESHOLD:
        ct.trigger = True

    if not ct.trigger:
        attack_buffer.append(magnitude)

    if ct.trigger:
        decay_buffer.append(magnitude)
        ct.increment()
        if ct.counter == DECAY:
            arr = np.concatenate([attack_buffer, decay_buffer])
            #====
            # Classify
            #====
            x_inference = torch.FloatTensor(arr).unsqueeze(0)
            x_inference = x_inference.unsqueeze(1)
            with torch.no_grad():  # Disable gradient computation for efficiency
                predictions = model(x_inference)
            predicted_class = torch.argmax(predictions, dim=1).item()
            print(surfaces[predicted_class])
            ct.reset()
            
    
    # print(f"Bang: {magnitude}")

model = Simple1DCNN1()
model = torch.load('model.pth', weights_only=False)
model.eval()
 
device = ximu3.PortScanner.scan()[0]
connection_info = device.connection_info
connection = ximu3.Connection(connection_info)

print(connection_info.to_string())
print(device.device_name)

connection.add_inertial_callback(inertial_callback)
#connection.add_earth_acceleration_callback(earth_acceleration_callback)
connection.open()

input("press enter to quit")

plt.plot(arr)
plt.show()

connection.close()
