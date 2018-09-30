# MELD
MELD: Multimodal EmotionLines Dataset for Emotion Recognition in Conversation

Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1300 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.


Please visit https://affective-meld.github.io for more details.


# Paper

The paper explaining this dataset can be found - https://github.com/SenticNet/MELD/blob/master/MELD.pdf

# Raw data

Please visit - http://bit.ly/MELD-raw to download the raw data. Data are stored in .mp4 format and can be found in XXX.tar.gz files.

# Description of the .csv files

#### Column Specification
| Column Name  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sr No.       | Serial numbers of the utterances mainly for referencing the utterances in case of different versions or multiple copies with different subsets. The serial number '-1' implies that the utterance was edited out from the DVD version of the TV Show we used or were unable to  locate and hence all the attributes pertaining to that utterance stand void apart from Emotion/Sentiment. The users are free to include the utterance in their experiment if they are able to find the corresponding clipping in the version of the show they possess. |
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
| NN_Flag      | Indicating that the utterance was annotated as non-neutral in the original dataset and relabelled.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
#### The files
- train_emotion.csv - contains the utterances in the training set along with Emotion labels and NN_flag.
- dev_emotion.csv - contains the utterances in the validation set along with Emotion labels and NN_flag.
- test_emotion.csv - contains the utterances in the test set along with Emotion labels and NN_flag.
- train_sent_emo.csv - contains the utterances in the training set along with Sentiment and Emotion labels.
- dev_sent_emo.csv - contains the utterances in the dev set along with Sentiment and Emotion labels.
- test_sent_emo.csv - contains the utterances in the test set along with Sentiment and Emotion labels.

###### Note: The utterances with serial no. '-1' are  only provided in the 'xxx_emotion.csv' files.

# Citation

Please cite the following papers if you find this dataset useful in your research 

S. Poria, D. Hazarika, N. Majumder, G. Naik, R. Mihalcea, E. Cambria. Multimodal EmotionLines:
A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. (2018)

Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W. EmotionLines: An Emotion Corpus of Multi-Party Conversations. arXiv preprint arXiv:1802.08379 (2018).
