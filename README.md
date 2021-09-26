# Movie genre prediction

Predict the genres of a movie based on the synopsis and year.

## Data

The input (i.e., training data) is expected to be a csv file with a header and comma-separated values.
The synopsis (also having that header name) for each sample is expected to be surrounded by double quotation marks.
The genres value can be a single genre or multiple genres separated by a space.
This is an example of a training file with a single sample:

```csv
movie_id,year,synopsis,genres
1,2006,"sample text",Action Adventure
```

## Usage

The following command will call the file with an input file.
It will do data preparation, make the model, train it on a training set and evaluate it using a validation set.
Last, it will print the _Mean Average Precision @ k_ score (for the top 5 predicted genres) on this validation set.

```bash
python train_predict.py data.csv
```
