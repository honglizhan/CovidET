# CovidET-EMNLP2022
This repo contains the dataset and code for our EMNLP 2022 paper. If you use this dataset, please cite our paper.

Title: Why Do You Feel This Way? Summarizing Triggers of Emotions in Social Media Posts

Authors: <a href="https://honglizhan.github.io/">Hongli Zhan</a>, <a href="https://www.tsosea.com/">Tiberiu Sosea</a>, <a href="https://www.cs.uic.edu/~cornelia/">Cornelia Caragea</a>, <a href="https://jessyli.com/">Junyi Jessy Li</a>

```bibtex
@inproceedings{zhan-etal-2022-feel,
    title = "Why Do You Feel This Way? Summarizing Triggers of Emotions in Social Media Posts",
    author = "Zhan, Hongli  and
      Sosea, Tiberiu  and
      Caragea, Cornelia  and
      Li, Junyi Jessy",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.642",
    pages = "9436--9453",
    abstract = "Crises such as the COVID-19 pandemic continuously threaten our world and emotionally affect billions of people worldwide in distinct ways. Understanding the triggers leading to people{'}s emotions is of crucial importance. Social media posts can be a good source of such analysis, yet these texts tend to be charged with multiple emotions, with triggers scattering across multiple sentences. This paper takes a novel angle, namely, emotion detection and trigger summarization, aiming to both detect perceived emotions in text, and summarize events and their appraisals that trigger each emotion. To support this goal, we introduce CovidET (Emotions and their Triggers during Covid-19), a dataset of {\textasciitilde}1,900 English Reddit posts related to COVID-19, which contains manual annotations of perceived emotions and abstractive summaries of their triggers described in the post. We develop strong baselines to jointly detect emotions and summarize emotion triggers. Our analyses show that CovidET presents new challenges in emotion-specific summarization, as well as multi-emotion detection in long social media posts.",
}
```

For legal concerns, we only release the annotations and the corresponding IDs in Reddit (and *not* original posts). <del>We recommend using the the <a href="https://psaw.readthedocs.io/en/latest/">PSAW wrapper for Pushshift API</a> to gather the original Reddit posts from the Reddit IDs in the dataset. We provide the ready-to-use script for scraping the original Reddit posts from the Reddit IDs in *retrieve_posts.ipynb*.</del> Note that for privacy issues, we also anonymize the names of people as well as businesses mentioned in our dataset.

## Update Feb 9, 2023
PushShift Service is Down and users are no longer able to retrieve all submissions prior to November 2022 ([source](https://www.reddit.com/r/pushshift/comments/10utf1d/first_day_trying_to_scrape_reddit_bad_timing/)). If you want to retrieve the original Reddit posts, please email me at [honglizhan@utexas.edu](mailto:honglizhan@utexas.edu) and I will share the dataset containing the original Reddit posts with you.

# Abstract
Crises such as the COVID-19 pandemic continuously threaten our world and emotionally affect billions of people worldwide in distinct ways. Understanding the triggers leading to people's emotions is of crucial importance. Social media posts can be a good source of such analysis, yet these texts tend to be charged with multiple emotions, with triggers scattering across multiple sentences. This paper takes a novel angle, namely, *emotion detection and trigger summarization*, aiming to both detect perceived emotions in text, and summarize events that trigger each emotion. To support this goal, we introduce CovidET (**E**motions and their **T**riggers during **Covid**-19), a dataset of ~1,900 English Reddit posts related to COVID-19, which contains manual annotations of perceived emotions and abstractive summaries of their triggers described in the post. We develop strong baselines to jointly detect emotions and summarize emotion triggers. Our analyses show that CovidET presents new challenges in emotion-specific summarization, as well as multi-emotion detection in long social media posts.


# Code
To use the code, please first expand the Json files in the `train_val_test` directory by adding a `Post` key (along with the Reddit post text obtained using the PSAW wrapper) to each entry.

**Emotion Detection**:

```bash
$ TOKENIZERS_PARALLELISM=false python emotion_detection.py \
	--emotion <emotion> \
	--training_path <...> \
	--validation_path <...> \
	--test_path <...> \
	--model bert-large-uncased \
	--batch_size <...> \
	--gradient_accumulation_steps <...> \
	--results_detection <filename> \
	--learning_rate <...>
```

**Summarization**:

```bash
$ TOKENIZERS_PARALLELISM=false python emotion_summarization.py \
	--emotion <emotion> \
	--training_path <...> \
	--validation_path <...> \
	--test_path <...> \
	--model facebook/bart-large-cnn \
	--batch_size <...> \
	--gradient_accumulation_steps <...> \
	--results_summarization <filename> \
	--learning_rate <...>
```

**Detection and Summarization**:

```bash
$ TOKENIZERS_PARALLELISM=false python detection_summarization.py \
	--emotion <emotion> \
	--training_path <...> \
	--validation_path <...> \
	--test_path <...> \
	--model facebook/bart-large-cnn \
	--batch_size <...> \
	--gradient_accumulation_steps <...> \
	--results_detection_summarization <filename> \
	--learning_rate <...>
```

[![RevolverMaps Live Traffic Map](https://rf.revolvermaps.com/w/3/s/a/7/0/0/ffffff/010020/aa0000/53advquazq5.png)](https://www.revolvermaps.com/livestats/53advquazq5/)

[<a href='https://clustrmaps.com/site/1btfs'  title='Visit tracker'><img src='//clustrmaps.com/map_v2.png?cl=080808&w=a&t=n&d=TmKJlDUPul9hB0enjLETw4XdPi96wFo-ZvRBcVkaRNs&co=ffffff&ct=808080'/></a>](https://clustrmaps.com/site/1btfs)