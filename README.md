# Fake-News-Detection-Web

1. Download the model parameters, dataset and Glove embeddings:
   https://drive.google.com/drive/folders/1imN6Z_i5WdhfoOWvz90BdJv48B7m9UfX?usp=sharing
   
2. Put all file into the same folder

3. Run the code by commend:
   ```js
   gunicorn -w 4 -b 0.0.0.0:8888 demo_web:server
   ```
