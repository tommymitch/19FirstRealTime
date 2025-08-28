import ximu3
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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

class Counter:
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

ct = Counter()

arr = np.array([])

attack_buffer = CircularBuffer(ATTACK)
decay_buffer = CircularBuffer(DECAY)


def inertial_callback(message):
    accelerometer = np.array([message.accelerometer_x, message.accelerometer_y, message.accelerometer_z])
    magnitude = np.linalg.norm(accelerometer)

    global arr

    if not ct.trigger and magnitude > THRESHOLD:
        ct.trigger = True
        print(f"Triggered: {magnitude}")

    if not ct.trigger:
        attack_buffer.append(magnitude)

    if ct.trigger and ct.counter < DECAY:
        decay_buffer.append(magnitude)
        ct.increment()
        if ct.counter == DECAY:
            print(f"Complete: {magnitude}")
            arr = np.concatenate([attack_buffer, decay_buffer])
            # ct.reset()
            
    
    # print(f"Bang: {magnitude}")
 
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
