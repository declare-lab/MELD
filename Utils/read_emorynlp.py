import pandas as pd
df_train = pd.read_csv('./emorynlp_test_final.csv') # load the .csv file, specify the appropriate path
utt = df_train['Utterance'].tolist() # load the list of utterances
sea = df_train['Season'].tolist() # load the list of season no.
ep = df_train['Episode'].tolist() # load the list of episode no.
sc_id = df_train['Scene_ID'].tolist() # load the list of scene id's
utt_id = df_train['Utterance_ID'].tolist() # load the list of utterance id's
for i in range(len(utt)):
    print ('Utterance: ' + utt[i]) # display utterance
    print ('Video Path: emorynlp_train_splits/sea' + str(sea[i]) + '_ep' + str(ep[i]) + '_sc' + str(sc_id[i]) + '_utt' + str(utt_id[i]) + '.mp4') # display the video file path
    print ()