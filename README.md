# DSCI303FinalProject
Final Project for DSCI 303
Deployed Web App Link: https://tinyurl.com/wrsbwcx

Members:

* Daniel Tang - Scrum Lead, Full Stack
* Patrick Han - Data Science Geek

Instructions for Front End: Open app.py in PyCharm or desired IDE and do ```Flask Run``` in frontend directory in terminal. This should load the front end.

Our Dataset: https://www.yelp.com/dataset/download, we only need "review.json"

For trimming the data
* Goto terminal and type:
  ```
  HEAD -10 review.json > review10lines.json
  ```
  to write the first ten reviews of review.json into a smaller json file. review.json is too big for the JsonParser.pynb to handle due to memory contraints of personal laptops. Make sure you change the filenames in JsonParser, the first block reads, the second writes.
* To run jupyter notebook type in terminal if you don't have jupyter notebook installed:
  ```
  pip install jupyter
  ```
  Once installed to run jupyter do, in directory you want to use:
  ```
  jupyter notebook
  ```

