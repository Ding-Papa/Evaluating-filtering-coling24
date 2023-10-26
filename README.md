# Evaluating-filtering-coling24
Code and prompt templates for evaluation-filtering
+ Some data were not uploaded due to size restrictions, but you can find all the datasets covered in this paper by the references in the paper.

You can run the *main_framework.py* file with additional arguements：
+ use *--dname datasetname* to specify the dataset for training and testing, and this will also create a new directory under your current path. The directory name is the name of the dataset, e.g. *HacRED*.
+ use *--do_train* to train an entity-extraction model for the dataset.
+ use *--cons_candidate* to cauculate the candidate entity pairs and output a candidate file in your *datasetname* directory. It is recommended to set the *--filter* parameter to *True* when generating candidates, that is, use the evaluation model proposed in our papaer to extract generate high-precision candidate pairs.
+ use *--do_eval* to calculate the metrics (precision, recall and F1 score) for the results. Please specify the file path of candidates and results in the line 319-324 of *main_framework.py*. The results files are generated by run the *llm_batch_inference.py* file.

You can run the *relcombtrainer.py* file to train and save our evaluation model:
+ use *--dname datasetname* to specify the dataset for training, and the evaluation model will be saved in the *datasetname* directory.

Our instructions and prompts are in the *llm_batch_inference.py* file, and you can run this file to perform our evaulation-filtering LLM-based extraction framework:
+ use *--dname datasetname* to specify the dataset for testing. Please specify the file path of candidates and save path of results in the line 168-190 of this code file.
+ use *--model* to specify the LLM model name. Please modify the model path in the line 142-144 of this code file.
+ use *--peft* to specify whether use peft model or not.
