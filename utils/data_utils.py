
import os 
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.utils as utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ConcatCohorts(imagesPath, cliniTablePath, slideTablePath,  label, reportFile, outputPath, csvName, patientNumber = 'ALL', minNumberOfTiles = 0):

    patients = []
    
    slideTableList = []
    clinicalTableList = []
    
    imgsList = []

    patientList = []
    slideList = []
    slideAdr = []
    labelList = []

    labelList_return = []
    patientList_return = []
    
    for imgCounter in range(len(imagesPath)):
                
        print('LOADING DATA FROM ' + imagesPath[imgCounter] + '...\n')
        reportFile.write('LOADING DATA FROM ' + imagesPath[imgCounter] + '...' + '\n')
        
        imgPath = imagesPath[imgCounter]
        cliniPath = cliniTablePath[imgCounter]
        slidePath = slideTablePath[imgCounter]
        
        if cliniPath.split('.')[-1] == 'csv':
            clinicalTable = pd.read_csv(cliniPath)
        else:
            clinicalTable = pd.read_excel(cliniPath)
        
        if slidePath.split('.')[-1] == 'csv':
            slideTable = pd.read_csv(slidePath)
        else:
            slideTable = pd.read_excel(slidePath)
        clinicalTable['label'] = clinicalTable['label'].replace(' ', '')
        lenBefore = len(clinicalTable)
        clinicalTable = clinicalTable[clinicalTable['label'].notna()]
        
        notAcceptedValues = ['NA', 'NA ', 'NAN', 'N/A', 'na', 'n.a', 'N.A', 'UNKNOWN', 'x', 'NotAPPLICABLE', 'NOTPERFORMED',
                             'NotPerformed', 'Notassigned', 'excluded', 'exclide', '#NULL', 'PerformedButNotAvailable', 'x_', 'NotReported', 'notreported', 'INCONCLUSIVE', 'Unknown']
        
        for i in notAcceptedValues:
            clinicalTable = clinicalTable[clinicalTable['label'] != i]

        lenafter = len(clinicalTable)
        
        print('Remove the NaN values from the Target Label...\n')
        print('{} Patients didnt have the proper label for target label: {}\n'.format(lenBefore - lenafter, label))
        reportFile.write('{} Patients didnt have the proper label for target label: {}'.format(lenBefore-lenafter, label) + '\n')
        
        clinicalTable_Patient = list(clinicalTable['PATIENT'])
        clinicalTable_Patient = list(set(clinicalTable_Patient))
        
        slideTable_Patint = list(slideTable['PATIENT'])
        slideTable_Patint = list(set(slideTable_Patint))

        inClinicalNotInSlide = []
        for item in clinicalTable_Patient:
            if not item in slideTable_Patint:
                inClinicalNotInSlide.append(item)
                
        print('Data for {} Patients from Clini Table is not found in Slide Table!\n'.format(len(inClinicalNotInSlide)))
        reportFile.write('Data for {} Patients from Clini Table is not found in Slide Table!'.format(len(inClinicalNotInSlide)) + '\n')
        
        inSlideNotInClinical = []
        for item in slideTable_Patint:
            if not item in clinicalTable_Patient:
                inSlideNotInClinical.append(item)
                
        print('Data for {} Patients from Slide Table is not found in Clini Table!\n'.format(len(inSlideNotInClinical)))
        reportFile.write('Data for {} Patients from Slide Table is not found in Clini Table!'.format(len(inSlideNotInClinical)) + '\n')
                
        print('-' * 30)
        reportFile.write('-' * 30 + '\n')
        
        patienID_temp = []
        for item in clinicalTable_Patient:
            if item in slideTable_Patint:
                patienID_temp.append(item)    

        patientIDs = []
        for item in patienID_temp:
            if item in clinicalTable_Patient:
                patientIDs.append(item)
        
        patientIDs = list(set(patientIDs))
        intersect = utils.intersection(patients, patientIDs)
        
        if not len(intersect) == 0:
            print(imagesPath[imgCounter])
            print(intersect)
            raise NameError('There are same PATIENT ID between COHORTS')
            
        imageNames = os.listdir(imgPath)
        imageNames =[os.path.join(imgPath, i) for i in imageNames]
        patients = patients + patientIDs
        imgsList = imgsList + imageNames
        
        clinicalTable = clinicalTable.loc[clinicalTable['PATIENT'].isin(patientIDs)]
        slideTable = slideTable.loc[slideTable['PATIENT'].isin(patientIDs)]
        
        clinicalTableList.append(clinicalTable[['PATIENT', 'label']])
        slideTableList.append(slideTable)
    
    clinicalTableList = pd.concat(clinicalTableList)
    slideTableList = pd.concat(slideTableList)
    
    slideTable_PatintNotUnique = list(slideTableList['PATIENT'])
    
    if patientNumber == 'ALL':        
        for patientID in tqdm(patients):            
            indicies = [i for i, n in enumerate(slideTable_PatintNotUnique) if n == patientID]
            matchedSlides = [list(slideTableList['FILENAME'])[i] for i in indicies] 
    
            temp = clinicalTableList.loc[clinicalTableList['PATIENT'] == str(patientID)]
            temp.reset_index(drop = True, inplace=True)
            
            for slide in matchedSlides:
                slideName = [i for i in imageNames if slide in i]
                if not len(slideName) == 0:
                    slideName = slideName[0]                         
                    if not len(os.listdir(slideName)) <= minNumberOfTiles:
                        patientList.append(patientID)
                        slideList.append(slideName.split('\\')[-1])
                        slideAdr.append(slideName) 
                        labelList.append(temp['label'][0])
                        if not patientID in patientList_return:
                            patientList_return.append(patientID)
                            labelList_return.append(temp['label'][0])
                else:
                    reportFile.write('Slide {} is dropped out because of Pre-Processing.'.format(slide) + '\n')
                
    else:
        patientsCopy = patients.copy()
        while len(patientList_return) < int(patientNumber):
            samplePatient = random.sample(patientsCopy, 1)[0]
            patientsCopy.remove(samplePatient)      
            indicies = [i for i, n in enumerate(slideTable_PatintNotUnique) if n == samplePatient]
            matchedSlides = [list(slideTableList['FILENAME'])[i] for i in indicies] 
            
            temp = clinicalTableList.loc[clinicalTableList['PATIENT'] == str(samplePatient)]
            temp.reset_index(drop = True, inplace=True)
             
            for slide in matchedSlides:
                slideName = [i for i in imageNames if slide in i]
                if not len(slideName) == 0:
                    slideName = slideName[0]                         
                    if not len(os.listdir(slideName)) < minNumberOfTiles:
                        patientList.append(samplePatient)
                        slideList.append(slideName.split('\\')[-1])
                        slideAdr.append(slideName) 
                        labelList.append(temp['label'][0])
                        if not samplePatient in patientList_return:
                            patientList_return.append(samplePatient)
                            labelList_return.append(temp['label'][0])
                else:
                    reportFile.write('Slide {} is dropped out because of Pre-Processing.'.format(slide) + '\n')
                                                                       
    data = pd.DataFrame()
    data['PATIENT'] = patientList
    data['FILENAME'] = slideList
    data['SlideAdr'] = slideAdr    
    data['label'] = labelList
             
    data.to_csv(os.path.join(outputPath, csvName + '.csv'),  index = False)
    return patientList_return, labelList_return, os.path.join(outputPath, csvName + '.csv')


def GetTiles(csvFile, label, maxBlockNum, target_labelDict, test = False, seed = 23, filterPatients = []):

    np.random.seed(seed)
    data = pd.read_csv(csvFile)
    
    if not len(filterPatients) == 0:
        patientsUnique = filterPatients
    else:
        patientsUnique = list(set(data['PATIENT']))        
    
    tilesPathList = []
    yTrueList = []
    yTrueLabelList = []
    patinetList = []
    
    for index, patientID in enumerate(tqdm(patientsUnique)):
        selectedData = data.loc[data['PATIENT'] == patientID]
        selectedData.reset_index(inplace = True)
        tempTiles = []
        for item in range(len(selectedData)):
            tempTiles.extend([os.path.join(selectedData['SlideAdr'][item], i) for i in os.listdir(selectedData['SlideAdr'][item])])
        if len(tempTiles) > maxBlockNum:   
            random.shuffle(tempTiles)
            tempTiles = np.random.choice(tempTiles, maxBlockNum, replace=False)
        for tile in tempTiles:
            tilesPathList.append(tile)
            yTrueList.append(utils.get_value_from_key(target_labelDict, selectedData['label'][0]))
            yTrueLabelList.append(selectedData['label'][0])
            patinetList.append(str(patientID))
                
    df = pd.DataFrame(list(zip(patinetList, tilesPathList, yTrueList, yTrueLabelList)), columns =['PATIENT', 'TilePath', 'yTrue', 'yTrueLabel'])     
    df_temp = df.dropna()
    
    if test:
        dfFromDict = df_temp
    else:            
        tags = list(df_temp['yTrue'].unique())
        tagsLength = []
        dfs = {}
        for tag in tags:
            temp = df_temp.loc[df_temp['yTrue'] == tag]
            temp = temp.sample(frac=1).reset_index(drop=True)
            dfs[tag] = temp 
            tagsLength.append(len(df_temp.loc[df_temp['yTrue'] == tag]))
        
        minSize = np.min(tagsLength)
        keys = list(dfs.keys())
        frames = []
        for key in keys:
            temp_len = len(dfs[key])
            diff_len = temp_len - minSize
            drop_indices = np.random.choice(dfs[key].index, diff_len, replace = False)
            frames.append(dfs[key].drop(drop_indices))
            
        dfFromDict = pd.concat(frames)
                    
    return dfFromDict



class DatasetLoader(torch.utils.data.Dataset):

    def __init__(self, imgs, labels, imgs2, labels2):
        self.labels = labels
        self.imgs = imgs
        self.labels2 = labels2
        self.imgs2 = imgs2
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        X = Image.open(self.imgs[index])
        y = self.labels[index]
        X2 = Image.open(self.imgs2[index])
        y2 = self.labels2[index]

        if self.transform is not None:
            X = self.transform(X)
            X2 = self.transform(X2)
        return X, y, X2, y2

