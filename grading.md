# Jonas (11.5/15P)

## Gathering Tracking Data (4.5/5P)
* data is logged correctly
    * yep, but the first row is '0,0,0,,,' in each file and therefore no 1000 samples for the 10s (1.5P)
* log files are named and structured appropiately 
    * yep (1P)
* logging can be started with the DIPPID device
    * yep (1P)
* enough data sets captured
    * yep (1P)

## Acticity Recognition (8/10P)
* the program loads training data correctly
    * nope, we had to download the data, rename paths (0.5P)
* training data is pre-processed appropiately
    * yep (2P)
* a classifier is trained with this training data when the program is started
    * yep (1P)
* the classifier recognizes activities correctly
    * not really, it's really jumpy (pun intended) between all activies and doesn't reset when doing nothing (1.5P)
* prediction accuracy for a test data set is printed
    * yep (1P)
* prediction works continously without requiring intervention by the user
    * yep (1P)
* the fitness training application works and displays training activities and if they are executed correctly
    * yep, but pictograms would've been nice (1P)


## General Code Quality (-1P)
* no requirements.txt
* fonts don't work, we had to debug
* program can't be closed with 'q' nor 'esc' (window closes, but terminal still running)
