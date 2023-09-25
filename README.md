# Fake-News-Detection-Web

1. Download the model parameters, dataset:
   https://drive.google.com/drive/folders/1imN6Z_i5WdhfoOWvz90BdJv48B7m9UfX?usp=sharing
   
2. Download glove embedding (glove.6B.300d.txt) and put it in a folder named embedding:
   https://nlp.stanford.edu/projects/glove/

3. Create conda environment:
   ```js
   conda env create -f enviroment.yaml
   ```

4. Run the code by command:
   ```js
   gunicorn -w 4 -b 0.0.0.0:8888 demo_web:server
   ```
