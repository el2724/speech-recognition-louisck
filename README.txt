
Dependencies:
numpy, scipy, scikit-learn

Also uses a submodule: jameslyons/python_speech_features
for MFCC extraction

After cloning the repo, initialize the submodule:
git submodule init
git submodule update

this will fill the python_speech_features submodule

To run:
python index.py > myresults.txt

You may easily edit the file to change the settings that it is run at
(look under if __name__ == '__main__' in index.py)

