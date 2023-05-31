WARNING: Windows PowerShell was found to work better than CMD 
-------------------------------------------------------------------------------------
HOW TO INSTALL
-------------------------------------------------------------------------------------
0- ----->MAKE SURE YOU ARE ABLE TO USE CONDA <-----
1- Clone or Download the repository
2- Open a terminal within the root directory
3- Run the following command to install pip "conda install pip"
4- Install the necessary libraries using the following commands:
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
	python -m pip install "tensorflow<2.11"
				(Tensorflow 2.11 and higher does not work. | 2.10 is recommended)
	pip install opencv-python
	pip install flask
	conda install numpy matplotlib 
5- Once finished, run the following command in the terminal "python .\server.py"
6- Let the neural network training run
7- Loss/Accuracy plots pop up, watch them to see how the neural network perform and close them so the server can start
8- Once the server starts, CTRL + Click the IP that shows to open a browser with the website
9- Enjoy!
----------------------------------------------------------------------------------------

--->  If interested in watching a demo of the program running, watch the WorkingDemo.mp4 file.  <---
