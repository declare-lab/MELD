import pandas as pd
df_train = pd.read_csv('./train_sent_emo.csv') # load the .csv file, specify the appropriate path
utt = df_train['Utterance'].tolist() # load the list of utterances
dia_id = df_train['Dialogue_ID'].tolist() # load the list of dialogue id's
utt_id = df_train['Utterance_ID'].tolist() # load the list of utterance id's
for i in range(len(utt)):
    print ('Utterance: ' + utt[i]) # display utterance
    print ('Video Path: train_splits/dia' + str(dia_id[i]) + '_utt' + str(utt_id[i]) + '.mp4') # display the video file path
    print ()