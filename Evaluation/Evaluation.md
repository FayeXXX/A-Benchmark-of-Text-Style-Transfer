### Evaluation

TSTBench utilizes both a uniform set of automatic metrics and human evaluation to systematically evaluate different TST algorithms in terms of style strength, content preservation, and fluency. 

* Automatic evaluation

  Firstly, train the evaluation model in the directory content_preservtion, fluency, style_acc respectively, then load the model and run eval_all.py.

* Human evaluation

  We measure the style strength, content preservation, fluency, and overall performance of the generated sentences by inviting three human annotators, all of whom are labmates with master's degrees or higher, to assign scores on a scale of 1 to 5 for each aspect. In total, 50 samples from each algorithm are randomly selected and given to three evaluators.  The scoring results are available in the directory human_average.