from PIL import Image
import pandas as pd

df = pd.read_csv('hc18_dataset/training_set_pixel_size_and_HC.csv')

for idx, row in df.iterrows():
    filename = row['filename']
    img_path = f'hc18_dataset/training_set/training_set/{filename}'
    img = Image.open(img_path)
    
    if 'aspect' not in img.info:
        print(f"Missing aspect metadata: {filename}")
        print(f"  Pixel size from CSV: {row['pixel size']}")
        print(f"  Image info: {img.info}")
