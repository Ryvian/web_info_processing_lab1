file_path = 'results/submission_weight_1.csv'
new_path = 'results/submission_weight_1_new.csv'

with open(file_path) as f:
    with open(new_path, 'w') as fw:
        for line in f:
            if line != '\n':
                fw.write(line)