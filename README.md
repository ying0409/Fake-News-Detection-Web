# Fake-News-Detection-Web

## About
This is a demo website for the paper: "Style-Aware Fack News Detection".
The model uses the construction of a word graph to capture the writing style of news, thus helping to discern the authenticity of the news.

## Demo Website
Link: http://140.113.24.124:8887/

## Reproduce
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
