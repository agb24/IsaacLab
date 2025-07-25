import torch

class GPURingBuffer:
    def __init__(self, capacity=100, dim=7, device='cuda:0'):
        # Init Params
        self.capacity = capacity
        self.dim = dim
        self.device = device
        # Create a Buffer
        self.buffer = torch.empty((capacity, dim), device=device)
        # Pointer to record which is the start position
        self.start = 0
        # Current Size of the Buffer Queue
        self.size = 0

    def append(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            x = torch.tensor(x, device=self.device)
        assert x.shape[-1] == self.dim, "Dimension mismatch"

        # Set the index for inserting the next item into Buffer Queue
        insert_index = (self.start + self.size) % self.capacity
        self.buffer[insert_index] = x

        # Update the current size of the Buffer Queue
        if self.size < self.capacity:
            self.size += 1
        else:
            # Ensures FIFO behavior
            self.start = (self.start + 1) % self.capacity  

    def get(self):
        # Returns the Buffer Queue contents in order
        if self.size == 0:
            return torch.empty((0, self.dim), device=self.device)
        
        # Get the next element from the Queue
        indices = [(self.start + i) % self.capacity for i in range(self.size)]
        return self.buffer[indices]

    def clear(self):
        # Reset the buffer
        self.start = 0
        self.size = 0


if __name__ == "__main__":
    buffer = GPURingBuffer()
    buffer.append(torch.randn(7))  # append a 7-dim vector
    all_items = buffer.get()       # returns all items in order
