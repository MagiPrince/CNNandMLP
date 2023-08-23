import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

class CustomCNNOutputModelWithConfidence(Model):
    def __init__(self, input_shape, num_outputs, output_dim):
        super(CustomCNNOutputModelWithConfidence, self).__init__()
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.output_dim = output_dim
        self.build_model()

    def build_model(self):
        self.inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu')(self.inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        
        self.outputs_coords = []
        self.outputs_confidence = []
        for _ in range(self.num_outputs):
            x_dense = Dense(64, activation='relu')(x)
            output_coords = Dense(2, activation='linear')(x_dense)  # Output x and y coordinates
            output_confidence = Dense(1, activation='sigmoid')(x_dense)
            self.outputs_coords.append(output_coords)
            self.outputs_confidence.append(output_confidence)
        
        self.model = Model(inputs=self.inputs, outputs=self.outputs_coords + self.outputs_confidence)

    def call(self, inputs):
        return self.model(inputs)

# Create the custom model with CNN and confidence scores
input_shape = (64, 64, 3)  # Input image shape (64x64 with 3 channels)
num_outputs = 10
output_dim = 2  # Two elements for x and y coordinates

model = CustomCNNOutputModelWithConfidence(input_shape, num_outputs, output_dim)

# Define loss functions for each output
losses = ['mean_squared_error'] * num_outputs + ['binary_crossentropy'] * num_outputs

# Compile the model with the specified losses
model.compile(optimizer='adam', loss=losses)

# Generate example input data
num_samples = 2  # Small number of samples for illustration
X_train = np.random.rand(num_samples, *input_shape)
y_train_coords = [np.random.rand(num_samples, output_dim) for _ in range(num_outputs)]
y_train_confidence = [np.random.randint(0, 2, size=(num_samples, 1)) for _ in range(num_outputs)]

# Train the model
model.fit(X_train, y_train_coords + y_train_confidence, epochs=10, batch_size=2, verbose=1)

# Example inference
output_coords_and_confidence = model.predict(X_train)
print("Output coordinates and confidence scores:")
print(output_coords_and_confidence)
