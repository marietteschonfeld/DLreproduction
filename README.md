# DLreproduction

This was a 35 hour attempt of getting the LSTS Learning Where to Focus for Efficient Video Object Detection by Jiang ZhengKai et al working.
This repository only shows the work that went in translationg the Python 2 Linux code to Python 3 and Windows.
Python 2 is deprecated since January 2020, while the author published the paper December 2020. This is why we at first tried to translate it.
This github repository is only a small part of our efforts, see [our blog](https://medium.com/death-of-the-author/death-of-the-author-and-the-authors-responsibility-for-their-codebase-685ad0226931) for more effort getting the original code running.
The project was quickly deemed too large to translate in full within the 35 hours, also including many libraries that only compile on linux, like bbox (using cython).
Some workarounds are implemented, seemingly getting around the use of those libraries for now. 
The current status of the repository is not far, since a huge amount of time went into fixing library issues, which is not all visibile from the repository.
The visible effort is mostly in adaptations in both some library files and the TrainLSTS.py, where bit by bit we tried to uncomment and change the original code and debug line by line where the errors come from.

Since the project was deemed too large to translate (1.4 million lines total, originally written in python 2) and the large number of libraries issues, we instead tried to get to get it working on Linux, in both Google Collab and Google Cloud VM.
This effort was also deemed unsuccesful, since there were also many incompatibility issues due to old library versions and instructions from the author that simply do not work (such as calling "python file1 file2", while the files do not exist and incomplete information on libraries.). 
See our blog for a complete story of our efforts to try and reproduce this paper: https://medium.com/death-of-the-author/death-of-the-author-and-the-authors-responsibility-for-their-codebase-685ad0226931


[The repository on Google Colab is visible here](https://colab.research.google.com/drive/1Q3L6JcIbh7HnBFz_9jniFS2r5-6J-Xv-?usp=sharing)
The description of what problems we encountered there and on Google Cloud are shared in the blog.
