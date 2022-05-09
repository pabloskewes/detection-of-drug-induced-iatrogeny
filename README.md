# Drug Iatrogeny Detector 💊
##### Machine Learning Detector for Drug Iatrogeny for the CHU Lille
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://github.com/AndreisPurim/DrugIatrogenyDetector)

Created by: [**Nicolas Acevedo**](https://github.com/nicoacevedor), [**Pablo Aldana**](https://github.com/Paldana99), [**Marie Lenglet**](https://github.com/mlenglet), [**Andreis Purim**](https://github.com/AndreisPurim) and [**Pablo Skewes**](https://github.com/pabloskewes)

This is an older project for the M1 Elective "Inteligence Artificel et Santé" at the École Centrale de Lille. The objective was to implement something inovative at the Centre Hospitalier Universitaire de Lille (CHU Lille) combining health and IA. We decided to create a program that every time a doctor added a new prescription, it would check and detect if it is correct and/or can cause further problems for the patient.

Since the data (which can't be released publicly, sorry) we had access to were prescriptions of past patients which were CORRECT, we had to work with unsupervised learning, especially clustering. The report is in the repository. A few explicative images can also help:

![](explanation1.PNG)

_Image explaining the basic idea: the doctor adds the drug prescription and the program detects incorrect ones_

![](explanation2.PNG)

_Some clustering with Doliprame. Since this was done with a PCA, it makes senses the clusters look slightly weird._

![](explanation3.PNG)

_Word2Vec made with the Code ATC from the prescription texts_

Since it is not maintained anymore (and it depends on other resources exclusive to the CHU), I'm leaving this as a legacy project. 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
