# model for information retreival given a query
## Non Supervised Model
Based on Vector Space Model.

Packages required: `pip3 install numpy, jieba, pandas, numba, pathos`

You can use it with running `main_non_supervised_model.py` (I use python3.7).
You may lower `content_prop` to speed up the process (with worse result), and tuning `title_weight` to improve the result.
### Problems
- Complete TF-iDF table is too large (~330000 word items, 13000 documents). If we store it with simple matrix, it takes ~35GB. No memory. (however it is sparse). 
- Dict based version used the sparsity, store the TF-iDF table with a 2-D dict. However the performance is poor. The bottleneck may be that it needs to rebuild the vector for each document for each query.
- thinking about storing the complete matrix in files, then read them in turn (File based, to do yet). 
- We can also use inverse index to filter out the docs without any word item in the query.

### Performance & Results
- For dict_based version, with content_prop == 0 (i.e. don't use the content of doc at all), the F1 score of result is ~0.77 (`submission_weight_1.csv` in the website), and it takes about 8 hours. We can improve the result by increasing content_prop (but slower). Some data:
    - each query for complete TF_iDF data takes about 23 min.
    - each query for only doc title TF_iDF data takes about 51 sec.
    - each 10000 more word items in TF_iDF takes about 0.82 min more time for each query.
    - content has about 330000 word items in total.



## Supervised Model
// to do
