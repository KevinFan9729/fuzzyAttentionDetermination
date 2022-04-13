# fuzzyAttentionDetermination
We utilize fuzzy logic to analyze user gaze direction and head orientation to determine user attention. 

## Attention Determination
Head orientation and gaze direction are key indicators of human attention. This module determines the attention level of the user. The module is robust to determine if the user is looking at the primary subject with tilted head postures.

## Fuzzy Logic
We track the locations of pupils to determine gaze direction. Because eyes are typically small compared to other facial and postural features, the smaller features cause the determination of gaze direction with naïve boolean thresholding to be unreliable. Therefore, we utilize fuzzy logic to determine eye gaze direction. 
