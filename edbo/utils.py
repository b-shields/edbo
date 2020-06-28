# Imports

import time

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from .math_utils import pca
from .chem_utils import ChemDraw

# Wall-clock timer for testing

class timer:
    
    def __init__(self, name):
        
        self.start = time.time()
        self.name = name
        
    def stop(self):
        """
        Returns wall clock-time.
        """
        
        self.end = time.time()    
        print(self.name + ': ' + str(self.end - self.start) + ' s')

# Data handling class

class Data:
    """
    Class or defining experiment domains and pre-processing.
    """
    
    def __init__(self, data):
        self.data = data
        self.base_data = data
    
    def reset(self):
        self.data = self.base_data
    
    def clean(self):
        self.data = drop_single_value_columns(self.data)
        self.data = drop_string_columns(self.data)
        
    def drop(self, drop_list):
        self.data = remove_features(self.data, drop_list)
    
    def standardize(self, target='yield', scaler='minmax'):
        self.data = standardize(self.data, target, scaler=scaler)
        
    def PCA(self, target='yield', n_components=1):
        pps = pca(self.data.drop(target, axis=1), n_components=n_components)
        pps[target] = self.data[target]
        self.data = pps
        
    def uncorrelated(self, target='yield', threshold=0.7):
        self.data = uncorrelated_features(self.data, 
                                          target, 
                                          threshold=threshold)
    
    def visualize(self, experiment_index_value, svg=True):
        
        columns = self.base_data.columns.values
        smi_bool = ['SMILES' in columns[i] for i in range(len(columns))]
        index = self.base_data[self.base_data.columns[smi_bool].values]
        
        SMILES_list = index.iloc[experiment_index_value].values
        cd = ChemDraw(SMILES_list, ipython_svg=svg)
        
        try:
            entry = self.base_data[self.index_headers].iloc[[experiment_index_value]]
        except:
            entry = self.base_data.iloc[[experiment_index_value]]
            
        print('\n##################################################### Experiment\n\n',
              entry,
              '\n')
        
        return cd.show()
    
    def get_experiments(self, index_values):
        try:
            entries = self.base_data[self.index_headers].iloc[index_values]
        except:
            entries = self.base_data.iloc[index_values]
            
        return entries

# Remove columns with only a single value

def drop_single_value_columns(df):
    """
    Drop datafame columns with zero variance. Return a new dataframe.
    """
    
    keep = []
    for i in range(len(df.columns)):
        if len(df.iloc[:,i].drop_duplicates()) > 1:
            keep.append(df.columns.values[i])
            
    return df[keep]
    
# Remove columns with non-numeric entries
    
def drop_string_columns(df):
    """
    Drop dataframe columns with non-numeric values. Return a new dataframe.
    """
    
    keep = []
    for i in range(len(df.columns)):
        unique = df.iloc[:,i].drop_duplicates()
        keepQ = True
        for j in range(len(unique)):
            if type(unique.iloc[j]) == type(''):
                keepQ = False
                break
        if keepQ: keep.append(df.columns.values[i])
        
    return df[keep]
        
# Remove unwanted descriptors

def remove_features(df, drop_list):
    """
    Remove features from dataframe with columns containing substrings in
    drop_list. Return a new dataframe.
    """

    keep = []
    for column_name in list(df.columns.values):
        keepQ = True
        for substring in list(drop_list):
            if substring in column_name:
                keepQ = False
                break
        if keepQ: keep.append(column_name)
    
    return df[keep]

# Standardize
    
def standardize(df, target, scaler='standard'):
    """
    Standardize descriptors but keep target.
    """
    
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    if target != None:
        data = df.drop(target,axis=1)
    else:
        data = df.copy()
    
    out = scaler.fit_transform(data)
    
    new_df = pd.DataFrame(data=out, columns=data.columns)
    
    if target != None:
        new_df[target] = df[target]
    
    return new_df

# Select uncorrelated set of features
    
def uncorrelated_features(df, target, threshold=0.95):
    """
    Returns an uncorrelated set of features.
    """
    
    if target != None:
        data = df.drop(target,axis=1)
    else:
        data = df.copy()
    
    corr = data.corr().abs()
    keep = []
    for i in range(len(corr.iloc[:,0])):
        above = corr.iloc[:i,i]
        if len(keep) > 0: above = above[keep]
        if len(above[above < threshold]) == len(above):
            keep.append(corr.columns.values[i])
    
    data = data[keep]
    
    if target != None:
        data[target] = list(df[target])
    
    return data

# edbo bot

class bot:
    """
    Bot class to be used for helping users resolve issues.
    """
    
    def __init__(self, name='edbo bot'):
        
        self.name = name
        
    def talk(self, text):
        """Print out text."""
        
        print('\n' + self.name + ': ' + text)
        
    def get_response(self, question):
        """Ask a question and wait for a response."""
        
        self.talk(question)
        text = input('~ ')
        return text
        
    def parse(self, text, triggers, not_triggers):
        """Parse response text for triggers/not_triggers."""
        
        # Test for trigger substrings
        bool_ = [True if t in text.lower() else False for t in triggers]
        
        # Test for anti-trigger substrings
        bool_not = [True if t in text.lower() else False for t in not_triggers]
        
        if True in bool_ and True not in bool_not:
            return True
        elif True not in bool_ and True not in bool_not:
            return 'Resolve'
        else:
            return False
        
    def multi_parse(self, text, trigger_dict):
        """Parse text for a number of responses."""
        
        # Test for triggers in substrings
        triggered = []
        for key in trigger_dict:
            bool_ = [True if t in text.lower() else False for t in trigger_dict[key]]
            if True in bool_:
                triggered.append(key)
                
        return triggered
        
    def parse_respond(self, text, triggers, not_triggers, response, not_response):
        """Parse response text for triggers/not_triggers and then respond."""
        
        check = self.parse(text, triggers, not_triggers)
        
        if check == 'Resolve':
            return 'Resolve'
        elif check:
            return response()
        else:
            return not_response()
        
    def resolve(self, question, question_root, triggers, not_triggers, response, not_response):
        """Resolve a boolean issue"""
        
        # Ask initial question and get response
        text = self.get_response(question)
        
        # Parse the response text        
        out = self.parse_respond(text, triggers, not_triggers, response, not_response)
        
        # Resolve if necessary
        while str(type(out)) == str(type('Resolve')) and out == 'Resolve':
            text_ = self.get_response('I didn\'t understand, ' + question_root)
            out = self.parse_respond(text_, triggers, not_triggers, response, not_response)
            
        return out
    
    def resolve_direct(self, question, trigger_dict, response_dict, print_dict, confirm_dict):
        """Resolve an issue with triggers and responses defined in dicts"""
        
        # Ask initial question and get response
        text = self.get_response(question)
        
        # Parse the response text        
        triggered = self.multi_parse(text, trigger_dict)
        
        while len(triggered) != 1:
            
            if len(triggered) == 0:
                text = self.get_response('I\'m not sure I can help you with that, rephrase or check the documentation. To exit type "exit"')
                triggered = self.multi_parse(text, trigger_dict)
                
            else:
                question = 'Can you clarify: '
                for t in triggered:
                    question += t + ', '
                question += 'or exit?'
                text = self.get_response(question)
                triggered = self.multi_parse(text, trigger_dict)
        
        # See if the question needs confirmation
        if triggered[0] in confirm_dict:
            text = self.get_response(confirm_dict[triggered[0]])
                
            if 'y' in text.lower():
                # Trigger appropriate response
                if triggered[0] in print_dict:
                    self.talk(print_dict[triggered[0]])
                return response_dict[triggered[0]]()
            else:
                return None
        else:
            if triggered[0] in print_dict:
                self.talk(print_dict[triggered[0]])
            return response_dict[triggered[0]]()
        


