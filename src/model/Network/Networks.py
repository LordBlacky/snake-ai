class FeedForward:

    def __init__(self, initializer, optimizer, loss_function):
        self.initializer = initializer
        self.optimizer = optimizer
        self.loss = []
        self.loss_function = loss_function
        self.layers = []
        self.data_tensor = None
        self.target_tensor = None
        self.error_tensor = None

    def initialize(self):
        for layer in self.layers:
            layer.initialize(self.initializer)

    def forward(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        for layer in self.layers:
            self.data_tensor = layer.forward(self.data_tensor)
        loss_value = self.loss_function.forward(self.data_tensor, self.target_tensor)
        self.loss.append(loss_value)
        return loss_value

    def backward(self):
        self.error_tensor = self.loss_function.backward(self.target_tensor)
        for layer in reversed(self.layers):
            self.error_tensor = layer.backward(self.error_tensor)

    def update(self):
        for layer in self.layers:
            layer.update(self.optimizer)

    def test(self, data_tensor):
        self.data_tensor = data_tensor
        for layer in self.layers:
            self.data_tensor = layer.forward(self.data_tensor)
        return self.data_tensor

    def append_layer(self, layer):
        self.layers.append(layer)
