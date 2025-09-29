import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
train_data = pd.read_csv('runs_nano/pose/train15/train15_train.csv')
val_data = pd.read_csv('runs_nano/pose/train15/train15_val.csv')

# Plot the data with finer lines
plt.figure(figsize=(10, 6))
plt.plot(train_data['Step'], train_data['Value'], label='Train', linewidth=0.3, marker='o', markersize=2)
plt.plot(val_data['Step'], val_data['Value'], label='Validation', linewidth=0.3, marker='s', markersize=2)

# Label the axes and add a title
plt.xlabel('Epoch')
plt.ylabel('pose_loss')
plt.title('train15')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
