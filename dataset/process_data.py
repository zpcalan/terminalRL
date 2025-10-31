from datasets import load_dataset

dataset = load_dataset('csv', data_files='latest_verified.csv')
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
dataset.save_to_disk('./')
