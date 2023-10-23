huggingface-cli login --token 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 
cd ..
git clone https://github.com/UKPLab/sentence-transformers.git
cd sentence-transformers
pip install -e .
cd ../DSC-Believer
pip install -r requirements.txt
sh scripts/down-data.sh
python src/DSC-Believer/data/data_transform.py
python src/DSC-Believer/test/test_sentence_bert.py