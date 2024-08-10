# Dataset Preprocessing Steps

  

- Download data set **Amazon Instant Video (5-core)** - from UCSD: http://jmcauley.ucsd.edu/data/amazon/index_2014.html

- Download the Sentires package **English-Jar** from Github: https://github.com/evison/Sentires
- Download the **Sentire-lei** package from Github: https://github.com/lileipisces/Sentires-Guide. 
- Put folder named ``lei`` from **Sentire-lei** into the folder **English-Jar**, modify the **4.lexicon.linux** with proper folder path on line 65, 78, 94, 95.
- Run these commands one by one:
	- ``python lei/0.format.py``
	- ``java -jar thuir-sentires.jar -t pre -c lei/1.pre``
	- ``java -jar thuir-sentires.jar -t pos -c lei/2.pos``
	- ``cp lei/intermediate/pos.1.txt lei/intermediate/pos.2.txt``
	- ``java -jar thuir-sentires.jar -t validate -c lei/3.validate``
	- ``java -jar thuir-sentires.jar -t lexicon -c lei/4.lexicon.linux``
	- ``java -jar thuir-sentires.jar -t profile -c lei/5.profile``
	- ``python lei/6.transform.py``
	- ``python lei/7.match.py``
- Modify **Sentires-preprocessing.py** with the proper folder path. Run the command:
	- ``python Sentires-preprocessing.py``
