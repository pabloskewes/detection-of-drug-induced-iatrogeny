'''
-------------------------------------------------------------------------------------
                             NETTOGAGE POUR IMPORTATION
-------------------------------------------------------------------------------------

This module contains functions to filter (or fix) the original csv "securimed",
creating a new one, in order to enable (or improve) the data import (before 
data processing).

filter_incorrect_columns:
    Opens the original securimed file, scrolls through it and saves in a new file
    only those lines that have the correct number of columns (101). The new file
    is saved in the same folder in csv format.

'''

data_path = r'G:\Data\prevention_iatrogenie_centrale_2\securimed.csv'
save_name = r'G:\Data\prevention_iatrogenie_centrale_2\securimed2.csv'


def file_tolist(file, PRINT: bool = True, frec: int = 100000):
    # file_tolist: file bool int -> list(str)
    # imports data from file into python list
    if PRINT:
        print('Loading',file,'into python list')
    data = []
    k = 0
    for line in file:
        if PRINT:
            if k % frec == 0:
                print('loading line',k)
        data.append(line)
        k += 1
    if PRINT:
        print(len(data),'lines succesfully loaded')
    return data
        
# write_filtered_csv: writes new file deleting wrong lines.

def import_data(filename: str, savename: str) -> None:
    errorfile_name = r'G:\Data\prevention_iatrogenie_centrale_2\temporary_import_error.csv'
    correctedfile_name = r'G:\Data\prevention_iatrogenie_centrale_2\temporary_import_error.csv'
    NUMBER_OF_SEMICOLS = 100
    in_error = 0
    file = open(filename,'r')
    file2 = open(savename,'w')
    errorfile = open(errorfile_name, 'w')
    data = file_tolist(file,PRINT=False)
    for line in data:
        if line.count(';') == NUMBER_OF_SEMICOLS:
            file2.write(line)
        else:
            errorfile.write(line)
            in_error+=1
    print("File succesfully filtered.",in_error,"lines sent to treatment")
    file.close()
    file2.close()
    errorfile.close()

def treat_errors(savename: str) -> None:
    errorfile_name = r'G:\Data\prevention_iatrogenie_centrale_2\temporary_import_error.csv'
    errorfile_name2 = r'G:\Data\prevention_iatrogenie_centrale_2\temporary_import_error2.csv'
    NUMBER_OF_SEMICOLS = 100
    file = open(errorfile_name,'r')
    file2 = open(savename,'a')
    file3 = open(errorfile_name2,'w')
    NUMBER_OF_SLASHES = 2
    SIZE_OF_DATE = 10
    data = file_tolist(file,PRINT=True)
    buffer=''
    in_error_continued = 0
    corrected = 0
    for line in data:
        if line[0:SIZE_OF_DATE].count('/') == NUMBER_OF_SLASHES:
            if buffer != '':
                file3.write(buffer+'\n')
                in_error_continued += 1
            buffer=line.strip('\n')
        else:
            buffer+="$$$ "+line.strip('\n')
            if buffer.count(';') == NUMBER_OF_SEMICOLS:
                corrected += 1
                file2.write(buffer+'\n')
                buffer=''
    file.close()
    file2.close()
    print(corrected,'Lines sucessfully recuperated',in_error_continued,'dropped')

def write_filtered_csv(filename: str, savename: str, treat_errors: bool = False) -> None:
    import_data(filename, savename)
    if(treat_errors):
        treat_errors(savename)




def filter_incorrect_columns2(filename: str, savename: str):
    '''
    filter_incorrect_columns: str str -> void
    takes path of file to read and path to save a new file. Saves only those lines from first
    file that has the correct amount of columns (101 columns, which means 100 ";")
    '''
    file = open(filename,'r')
    file2 = open(savename,'w')
    data = file_tolist(file,PRINT=False)
    NUMBER_OF_SEMICOLS = 100 
    line_error = False 
    saved = 0
    deleted = 0
    buffer = ''
    for i in range(len(data)):
        buffer_semicol= buffer.count(';')
        if buffer_semicol >= NUMBER_OF_SEMICOLS:
            if buffer_semicol == NUMBER_OF_SEMICOLS:
                file2.write(buffer)
                saved+=1
            else:
                print("Erasing buffer because it has more semicolons than it should")
            buffer = ''
        semicol = data[i].count(';')
        if semicol == NUMBER_OF_SEMICOLS:
            if line_error:
                if buffer != '':
                    deleted+=1
                buffer = ''
                file2.write(buffer)
                saved+=1
            else:
                file2.write(buffer)
                saved+=1
            line_error = False
        else:
            if semicol > NUMBER_OF_SEMICOLS:
                splitted_line = data[i].split(';')
                completed_line = ';'.join(splitted_line[0:NUMBER_OF_SEMICOLS+1])
                incomplet_line = ';'.join(splitted_line[NUMBER_OF_SEMICOLS+1:])
                file2.write(buffer)
                saved+=1
                if buffer != '':
                    print("WARNING: Erasing buffer since line",i,"is too big")
                buffer = incomplet_line
            elif semicol < NUMBER_OF_SEMICOLS:
                if buffer != '' and buffer.count(';') != NUMBER_OF_SEMICOLS and not buffer.endswith(';'):
                    buffer += ';'
                buffer += data[i]
            else:
                print("WARNING: Strange behavior in the number of semicolons, apparently it is equal but it is being sent to the incomplete")    
            line_error = True
    total = len(data)
    print("File had",total,"lines")
    print(saved,"lines saved,",deleted,"deleted.",total,"-",saved+deleted,"=",total-(saved+deleted))
    file.close()
    file2.close()
