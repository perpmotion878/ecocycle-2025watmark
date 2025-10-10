import tensorflow as tf

# For 160x120 grayscale or color images; change 3 to 1 if grayscale
input_shape = (160, 120, 3)  # Use (160, 120, 1) for grayscale

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D()
    tf.keras.layers.Dense(2, activation='softmax')  # 2 output classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

---
import faulthandler
import signal

# List of signals for which to disable faulthandler breakpoints
signals_to_disable = [
    signal.SIGINT,   # Interrupt from keyboard (Ctrl+C)
    signal.SIGUSR1,  # User-defined signal 1
    signal.SIGUSR2,  # User-defined signal 2
    signal.SIGXFSZ   # File size limit exceeded
]

# Disable faulthandler for the specified signals
for sig in signals_to_disable:
    try:
        faulthandler.disable(sig)
        print(f"faulthandler disabled for signal: {sig.name}")
    except (AttributeError, RuntimeError, ValueError) as e:
        print(f"Could not disable faulthandler for {sig}: {e}")
