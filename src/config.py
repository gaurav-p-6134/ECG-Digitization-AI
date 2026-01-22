# System Configuration
IMAGE_SIZE = (1696, 4352)  # (Height, Width) input for the model
device = "cuda"            # Change to 'cpu' if no GPU available

# ECG Digitization Constants
VOLTAGE_RESOLUTION = 78.74  # Pixels per mV (calibrated)
ZERO_LEVELS = [703.5, 987.5, 1271.5, 1531.5]  # Y-coordinates of the 0mV line for each row
TIME_START = 235
TIME_END = 4161