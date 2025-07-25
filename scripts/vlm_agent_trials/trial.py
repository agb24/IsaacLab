import hid
import time

import pyspacemouse


def find_device():
    """Find the device connected to computer."""
    found = False

    this_device = hid.device()

    for device_dict in hid.enumerate():
        keys = list(device_dict.keys())
        keys.sort()
        for key in keys:
            print("%s : %s" % (key, device_dict[key]))
        print()

    # implement a timeout for device search
    for _ in range(5):
        for device in hid.enumerate():
            if (
                device["product_string"] == "SpaceMouse Compact"
                or device["product_string"] == "SpaceMouse Wireless"
            ):
                # set found flag
                found = True
                vendor_id = device["vendor_id"]
                product_id = device["product_id"]
                print(vendor_id, product_id)
                # connect to the device
                this_device.close()
                this_device.open(vendor_id, product_id)
                
        # check if device found
        if not found:
            time.sleep(1.0)
        else:
            break
    # no device found: return false
    if not found:
        raise OSError("No device found by SpaceMouse. Is the device connected?")


def alt_spacemouse():
    success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)
    if success:
        while 1:
            state = pyspacemouse.read()
            time.sleep(0.01)


'''
import torch

# Parameters
buffer_size = 10
tensor_shape = (2,)  # Shape of each action tensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the buffer on GPU
buffer = torch.zeros((buffer_size, *tensor_shape), device=device)
write_index = 0
current_size = 0

def add_tensor(tensor):
    global write_index, current_size
    buffer[write_index] = tensor.to(device)
    write_index = (write_index + 1) % buffer_size
    current_size = min(current_size + 1, buffer_size)

    print(buffer)

    if current_size >= 2:
        run_commands()

def run_commands():
    # Example: Use the last 2 added tensors
    indices = [(write_index - 2) % buffer_size, (write_index - 1) % buffer_size]
    selected = buffer[indices]
    print("Running commands on:", selected)
    # Here you'd perform your GPU-based processing on `selected`

# Add example tensors
add_tensor(torch.tensor([1.0, 2.0]))
add_tensor(torch.tensor([3.0, 4.0]))
'''


if __name__ == "__main__":
    find_device()
    
    #alt_spacemouse()