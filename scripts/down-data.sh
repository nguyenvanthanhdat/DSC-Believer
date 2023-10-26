mkdir data

cd data
# wget -nc 
gdown 1kCifEFEvL0GY77lY2XZZAm9esm04HtA9 
gdown 1kVugTUWy-ZC3q0xklSjQnXjUHixmeMfX 
# unzip data file
# tar -xf train_val_images.zip
python -m zipfile -e DSC-public.zip .
python -m zipfile -e DSC-public-retrieval.zip .
