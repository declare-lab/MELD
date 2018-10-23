# MELD v1.0
----------------------------------------------------

MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation


# Introduction
Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.

### Example Dialogue
![](https://github.com/SenticNet/MELD/blob/master/images/emotion_shift.jpeg)

### Dataset Statistics
| Statistics                      | Train   | Dev     | Test    |
|---------------------------------|---------|---------|---------|
| # of modality                   | {a,v,t} | {a,v,t} | {a,v,t} |
| # of unique words               | 10,643  | 2,384   | 4,361   |
| Avg. utterance length           | 8.03    | 7.99    | 8.28    |
| Max. utterance length           | 69      | 37      | 45      |
| Avg. # of emotions per dialogue | 3.30    | 3.35    | 3.24    |
| # of dialogues                  | 1039    | 114     | 280     |
| # of utterances                 | 9989    | 1109    | 2610    |
| # of speakers                   | 260     | 47      | 100     |
| # of emotion shift              | 4003    | 427     | 1003    |
| Avg. duration of an utterance   | 3.59s   | 3.59s   | 3.58s   |

Please visit https://affective-meld.github.io for more details.



# Purpose
Multimodal data analysis exploits information from multiple-parallel data channels for decision making. With the rapid growth of AI, multimodal emotion recognition has gained a major research interest, primarily due to its potential applications in many challenging tasks, such as dialogue generation, multimodal interaction etc. A conversational emotion recognition system can be used to generate appropriate responses by analysing user emotions. Although there are numerous works carried out on multimodal emotion recognition, only a very few actually focus on understanding emotions in conversations. However, their work is limited only to dyadic conversation understanding and thus not scalable to emotion recognition in multi-party conversations having more than two participants. EmotionLines can be used as a resource for emotion recognition for text only, as it does not include data from other modalities such as visual and audio. At the same time, it should be noted that there is no multimodal multi-party conversational dataset available for emotion recognition research. In this work, we have extended, improved, and further developed EmotionLines dataset for the multimodal scenario. Emotion recognition in sequential turns has several challenges and context understanding is one of them. The emotion change and emotion flow in the sequence of turns in a dialogue make accurate context modelling a difficult task. In this dataset, as we have access to the multimodal data sources for each dialogue, we hypothesise that it will improve the context modelling thus benefiting the overall emotion recognition performance.  This dataset can also be used to develop a multimodal affective dialogue system. IEMOCAP, SEMAINE are multimodal conversational datasets which contain emotion label for each utterance. However, these datasets are dyadic in nature, which justifies the importance of our Multimodal-EmotionLines dataset. The other publicly available multimodal emotion and sentiment recognition datasets are MOSEI, MOSI, MOUD. However, none of those datasets is conversational.

# Dataset Creation
The first step deals with finding the timestamp of every utterance in each of the dialogues present in the EmotionLines dataset. To accomplish this, we crawled through the subtitle files of all the episodes which contains the beginning and the end timestamp of the utterances. This process enabled us to obtain season ID, episode ID, and timestamp of each utterance in the episode. We put two constraints whilst obtaining the timestamps: (a) timestamps of the utterances in a dialogue must be in increasing order, (b) all the utterances in a dialogue have to belong to the same episode and scene.
Constraining with these two conditions revealed that in EmotionLines, a few dialogues consist of multiple natural dialogues. We filtered out those cases from the dataset. Because of this error correction step, in our case, we have the different number of dialogues as compare to the EmotionLines. After obtaining the timestamp of each utterance, we extracted their corresponding audio-visual clips from the source episode. Separately, we also took out the audio content from those video clips. Finally, the dataset contains visual, audio, and textual modality for each dialogue.

# Paper
The paper explaining this dataset can be found - https://arxiv.org/pdf/1810.02508.pdf

# Download the data
Please visit - http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz to download the raw data. Data are stored in .mp4 format and can be found in XXX.tar.gz files. Annotations can be found in https://github.com/SenticNet/MELD/data/MELD/.

# Description of the .csv files

#### Column Specification
| Column Name  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sr No.       | Serial numbers of the utterances mainly for referencing the utterances in case of different versions or multiple copies with different subsets |
| Utterance    | Individual utterances from EmotionLines as a string.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Speaker      | Name of the speaker associated with the utterance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Emotion      | The emotion (neutral, joy, sadness, anger, surprise, fear, disgust) expressed by the speaker in the utterance.                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Sentiment    | The sentiment (positive, neutral, negative) expressed by the speaker in the utterance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Dialogue_ID  | The index of the dialogue starting from 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Utterance_ID | The index of the particular utterance in the dialogue starting from 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Season       | The season no. of Friends TV Show to which a particular utterance belongs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Episode      | The episode no. of Friends TV Show in a particular season to which the utterance belongs.                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| StartTime    | The starting time of the utterance in the given episode in the format 'hh:mm:ss,ms'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| EndTime      | The ending time of the utterance in the given episode in the format 'hh:mm:ss,ms'.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

#### The files
- /data/MELD/train_sent_emo.csv - contains the utterances in the training set along with Sentiment and Emotion labels.
- /data/MELD/dev_sent_emo.csv - contains the utterances in the dev set along with Sentiment and Emotion labels.
- /data/MELD/test_sent_emo.csv - contains the utterances in the test set along with Sentiment and Emotion labels.

# Description of Raw Data
- There are 3 folders (.tar.gz files)-train, dev and test; each of which corresponds to video clips from the utterances in the 3 .csv files.
- In any folder, each video clip in the raw data corresponds to one utterance in the corresponding .csv file. The video clips are named in the format: diaX1\_uttX2.mp4, where X1 is the Dialogue\_ID and X2 is the Utterance_ID as provided in the corresponding .csv file, denoting the particular utterance.
- For example, consider the video clip **dia6_utt1.mp4** in **train.tar.gz**. The corresponding utterance for this video clip will be in the file **train_sent_emp.csv** with **Dialogue_ID=6** and **Utterance_ID=1**, which is *'You liked it? You really liked it?'*

# Labelling
For experimentation, all the labels are represented as one-hot encodings, the indices for which are as follows:
- **Emotion** - {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}. Therefore, the label corresponding to the emotion *'joy'* would be [0., 0., 0., 0., 1., 0., 0.]
- **Sentiment** - {'neutral': 0, 'positive': 1, 'negative': 2}. Therefore, the label corresponding to the sentiment *'positive'* would be [0., 1., 0.]

# Run the baseline

Please follow these steps to run the baseline - 

1. Download the features from [here](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Features.Models.tar.gz).
2. Copy these features into `./data/pickles/`
3. To train/test the baseline model, run the file: `baseline.py` as follows:
    - `python baseline.py -classify [Sentiment|Emotion] -modality [text|audio|bimodal] [-train|-test]` 
    - example command to train text unimodal for sentiment classification: `python baseline.py -classify Sentiment -modality text -train`
    - use `python baseline.py -h` to get help text for the parameters.
4. For pre-trained models, download the model weights from [here](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Features.Models.tar.gz) and place the pickle files inside `./data/models/`.

# Citation
Please cite the following papers if you find this dataset useful in your research

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. Multimodal EmotionLines: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. (2018)

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W. EmotionLines: An Emotion Corpus of Multi-Party Conversations. arXiv preprint arXiv:1802.08379 (2018).

# Additionaly we also release:
### Multimodal EmoryNLP Emotion Detection Dataset
#### Description
Multimodal EmoryNLP Emotion Detection Dataset has been created by enhancing and extending EmoryNLP Emotion Detection dataset. It contains the same dialogue instances available in EmoryNLP Emotion Detection dataset, but it also encompasses audio and visual modality along with text. It has more than 800 scenes and 9000 utterances from Friends TV series. Multiple speakers participated in the scenes. Each utterance in a scene has been labeled by any of these seven emotions -- Neutral, Joyful, Peaceful, Powerful, Scared, Mad and Sad. The annotations are borrowed from the original dataset.
#### Dataset Statistics
| Statistics                      | Train   | Dev     | Test    |
|---------------------------------|---------|---------|---------|
| # of modality                   | {a,v,t} | {a,v,t} | {a,v,t} |
| # of unique words               | 9,744  | 2,123   | 2,345   |
| Avg. utterance length           | 7.86    | 6.97    | 7.79    |
| Max. utterance length           | 78      | 60      | 61      |
| Avg. # of emotions per scene | 4.10    | 4.00    | 4.40    |
| # of scenes                  | 659    | 89     | 79     |
| # of utterances                 | 7551    | 954    | 984    |
| # of speakers                   | 250     | 46      | 48     |
| # of emotion shift              | 4596    | 575     | 653    |
| Avg. duration of an utterance   | 5.55s   | 5.46s   | 5.27s   |

#### Data
The dataset containing video files can be found [here](https://drive.google.com/open?id=1gmK8X3mAEV3Eubd6rSpIyOprrSYE5YZE).
The annotation files can be found in https://github.com/senticnet/MELD/data/emorynlp/. There are 3 .csv files. The .csv files are provided so that the user can find the particular utterance to its video clip. Each utterance video clip is indexed by the season no., episode no., scene id and utterance id. as given in the .csv files corresponding to a particular utterance. For example, **sea1\_ep2\_sc6\_utt3.mp4** implies the clip corresponds to the utterance with season no. 1, episode no. 2, scene\_id 6 and utterance\_id 3. The indexing is in agreement with the original dataset. The .csv files and the video files are divided into the train, validation and test set in accordance with the original dataset. Annotations have been directly borrowed from the original EmoryNLP dataset (Zahiri et al. (2018)).

***Note***: There are few utterances for which we were not able to find the start and end time due to its difference with the subtitles. Such utterances have been omitted from the dataset. However, the user is free to find the corresponding utterances from the original dataset and generate video clips for the same.
#### Citation
Please cite the following papers if you find this dataset useful in your research

S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. Multimodal EmotionLines: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. (2018)

S. Zahiri and J. D. Choi. Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks. In The AAAI Workshop on Affective Content Analysis, AFFCON'18, 2018.
