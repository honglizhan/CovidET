# CovidET-EMNLP2022
This repo contains the dataset and code for our EMNLP 2022 paper. If you use this dataset, please cite our paper.

Title: Why Do You Feel This Way? Summarizing Triggers of Emotions in Social Media Posts

Authors: <a href="https://honglizhan.github.io/">Hongli Zhan</a>, <a href="https://www.tsosea.com/">Tiberiu Sosea</a>, <a href="https://www.cs.uic.edu/~cornelia/">Cornelia Caragea</a>, <a href="https://jessyli.com/">Junyi Jessy Li</a>

```bibtex
@inproceedings{ZhanETAL22CovidET,
  author = {Zhan, Hongli and Sosea, Tiberiu and Caragea, Cornelia and Li, Junyi Jessy},
  title = {Why Do You Feel This Way? Summarizing Triggers of Emotions in Social Media Posts},
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  pages = {To appear},
  year = {2022},
}
```

For legal concerns, we only release the annotations and the corresponding IDs in Reddit (and *not* original posts). We recommend using the the <a href="https://psaw.readthedocs.io/en/latest/">PSAW wrapper for Pushshift API</a> to gather the original Reddit posts from the Reddit IDs in the dataset. We provide the ready-to-use script for scraping the original Reddit posts from the Reddit IDs in *retrieve_posts.ipynb*. Note that for privacy issues, we also anonymize the names of people as well as businesses mentioned in our dataset.

# Abstract
Crises such as the COVID-19 pandemic continuously threaten our world and emotionally affect billions of people worldwide in distinct ways. Understanding the triggers leading to people's emotions is of crucial importance. Social media posts can be a good source of such analysis, yet these texts tend to be charged with multiple emotions, with triggers scattering across multiple sentences. This paper takes a novel angle, namely, *emotion detection and trigger summarization*, aiming to both detect perceived emotions in text, and summarize events that trigger each emotion. To support this goal, we introduce CovidET (**E**motions and their **T**riggers during **Covid**-19), a dataset of ~1,900 English Reddit posts related to COVID-19, which contains manual annotations of perceived emotions and abstractive summaries of their triggers described in the post. We develop strong baselines to jointly detect emotions and summarize emotion triggers. Our analyses show that CovidET presents new challenges in emotion-specific summarization, as well as multi-emotion detection in long social media posts.


# Codes
To use the code, please first expand the Json files in the `train_val_test` directory by adding a `Post` key (along with the Reddit post text obtained using the PSAW wrapper) to each entry.

Emotion Detection:

```
TOKENIZERS_PARALLELISM=false python emotion_detection.py --emotion <emotion> --training_path <...> ---validation_path <...> --test_path <...> --model bert-large-uncased --batch_size <...> --gradient_accumulation_steps <...> --results_detection <filename> --learning_rate <...>
```

Summarization:

```
TOKENIZERS_PARALLELISM=false python emotion_summarization.py --emotion <emotion> --training_path <...> ---validation_path <...> --test_path <...> --model facebook/bart-large-cnn --batch_size <...> --gradient_accumulation_steps <...> --results_summarization <filename> --learning_rate <...>
```

Detection and Summarization:

```
TOKENIZERS_PARALLELISM=false python detection_summarization.py --emotion <emotion> --training_path <...> ---validation_path <...> --test_path <...> --model facebook/bart-large-cnn --batch_size <...> --gradient_accumulation_steps <...> --results_detection_summarization <filename> --learning_rate <...>
```
