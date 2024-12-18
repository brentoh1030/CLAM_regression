import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth, generate_split_no_cls

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		event_time_col='event_time',   # Add for survival task
        censorship_col='censorship',  # Add for survival task
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		#self.num_classes = len(set(self.label_dict.values()))
		self.num_classes = 2
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.event_time_col = event_time_col  # Save column name for event times
		self.censorship_col = censorship_col  # Save column name for censorship
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		if self.label_dict is not None:  # Only prepare class IDs for classification tasks
			# Store IDs corresponding to each class at the patient level
			self.patient_cls_ids = [[] for i in range(self.num_classes)]
			for i in range(self.num_classes):
				self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

			# Store IDs corresponding to each class at the slide level
			self.slide_cls_ids = [[] for i in range(self.num_classes)]
			for i in range(self.num_classes):
				self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id']))  # Get unique patients
		if self.label_dict is not None:  # For classification tasks
			patient_labels = []
			for p in patients:
				locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
				assert len(locations) > 0
				label = self.slide_data['label'][locations].values
				if patient_voting == 'max':
					label = label.max()  # Get patient label (MIL convention)
				elif patient_voting == 'maj':
					label = stats.mode(label)[0]
				else:
					raise NotImplementedError
				patient_labels.append(label)
			self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}
		else:  # For survival tasks
			self.patient_data = {'case_id': patients}  # No label processing required

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_dict is not None:  # Only process labels if label_dict is provided
			if label_col != 'label':
				data['label'] = data[label_col].copy()

			mask = data['label'].isin(ignore)
			data = data[~mask]
			data.reset_index(drop=True, inplace=True)
			for i in data.index:
				key = data.loc[i, 'label']
				data.at[i, 'label'] = label_dict[key]
				
		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)
	def summarize(self):
		if self.label_dict is not None:  # For classification tasks
			print("label column: {}".format(self.label_col))
			print("label dictionary: {}".format(self.label_dict))
			print("number of classes: {}".format(self.num_classes))
			print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
		else:  # For survival tasks
			print("This is a survival prediction task.")
			print("Event time stats:")
			print(self.slide_data[self.event_time_col].describe())
			print("Censorship stats:")
			print(self.slide_data[self.censorship_col].value_counts())

		#for i in range(self.num_classes):
		#	print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
		#	print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}
		print(f"Total samples: {len(self.slide_data)}, Val samples: {val_num}, Test samples: {test_num}")

		#if self.patient_strat:
		#	settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		#else:
		#	settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		if hasattr(self, 'patient_cls_ids') and self.patient_cls_ids:  # Only for classification tasks
			settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
			self.split_gen = generate_split(**settings)
		else:  # For survival tasks
			self.split_gen = generate_split_no_cls(
				samples = len(self.patient_data['case_id']) if self.patient_strat else len(self.slide_data), 
				n_splits=k, 
				val_num=val_num, 
				test_num=test_num, 
				label_frac=label_frac, 
				seed=self.seed, 
				custom_test_ids=custom_test_ids)

	

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
			print(f"Train IDs: {len(ids[0])}, Val IDs: {len(ids[1])}, Test IDs: {len(ids[2])}")

		else:
			self.train_ids, self.val_ids, self.test_ids = ids
			print(f"Train IDs: {len(ids[0])}, Val IDs: {len(ids[1])}, Test IDs: {len(ids[2])}")
		print(f"After set_splits: Train size: {len(self.train_ids)}, Val size: {len(self.val_ids)}, Test size: {len(self.test_ids)}")


	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):


		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
		print(f"Generated splits: Train size: {len(train_split)}, Val size: {len(val_split)}, Test size: {len(test_split)}")
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			if self.label_dict:
				# For classification tasks
				index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
				columns = ['train', 'val', 'test']
				df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index, columns=columns)
			else:
				# For survival tasks
				index = ['event_time', 'censorship']
				columns = ['train_stats', 'val_stats', 'test_stats']
				df = pd.DataFrame(columns=columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		if self.label_dict:
			labels = self.getlabel(self.train_ids)
			unique, counts = np.unique(labels, return_counts=True)
			for u in range(len(unique)):
				print(f'Number of samples in class {unique[u]}: {counts[u]}')
				if return_descriptor:
					df.loc[index[u], 'train'] = counts[u]
		else:
			print("Event time stats (Train):")
			print(self.slide_data.loc[self.train_ids, self.event_time_col].describe())
			print("Censorship stats (Train):")
			print(self.slide_data.loc[self.train_ids, self.censorship_col].value_counts())
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		if self.label_dict:
			labels = self.getlabel(self.val_ids)
			unique, counts = np.unique(labels, return_counts=True)
			for u in range(len(unique)):
				print(f'Number of samples in class {unique[u]}: {counts[u]}')
				if return_descriptor:
					df.loc[index[u], 'val'] = counts[u]
		else:
			print("Event time stats (Val):")
			print(self.slide_data.loc[self.val_ids, self.event_time_col].describe())
			print("Censorship stats (Val):")
			print(self.slide_data.loc[self.val_ids, self.censorship_col].value_counts())

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		if self.label_dict:
			labels = self.getlabel(self.test_ids)
			unique, counts = np.unique(labels, return_counts=True)
			for u in range(len(unique)):
				print(f'Number of samples in class {unique[u]}: {counts[u]}')
				if return_descriptor:
					df.loc[index[u], 'test'] = counts[u]
		else:
			print("Event time stats (Test):")
			print(self.slide_data.loc[self.test_ids, self.event_time_col].describe())
			print("Censorship stats (Test):")
			print(self.slide_data.loc[self.test_ids, self.censorship_col].value_counts())

		if return_descriptor and not self.label_dict:
		# Expand the descriptor DataFrame for survival tasks
			train_event_stats = self.slide_data.loc[self.train_ids, self.event_time_col].describe()
			val_event_stats = self.slide_data.loc[self.val_ids, self.event_time_col].describe()
			test_event_stats = self.slide_data.loc[self.test_ids, self.event_time_col].describe()

			censorship_train_stats = self.slide_data.loc[self.train_ids, self.censorship_col].value_counts()
			censorship_val_stats = self.slide_data.loc[self.val_ids, self.censorship_col].value_counts()
			censorship_test_stats = self.slide_data.loc[self.test_ids, self.censorship_col].value_counts()

			# Add event time statistics
			for stat_name, value in train_event_stats.items():
				df.loc[f'event_time_{stat_name}', 'train_stats'] = value
			for stat_name, value in val_event_stats.items():
				df.loc[f'event_time_{stat_name}', 'val_stats'] = value
			for stat_name, value in test_event_stats.items():
				df.loc[f'event_time_{stat_name}', 'test_stats'] = value

			# Add censorship statistics
			for censored_value, count in censorship_train_stats.items():
				df.loc[f'censorship_{censored_value}', 'train_stats'] = count
			for censored_value, count in censorship_val_stats.items():
				df.loc[f'censorship_{censored_value}', 'val_stats'] = count
			for censored_value, count in censorship_test_stats.items():
				df.loc[f'censorship_{censored_value}', 'test_stats'] = count




		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0
		
		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		#label = self.slide_data['label'][idx]
		event_time = self.slide_data[self.event_time_col][idx]  # Get event time
		censorship = self.slide_data[self.censorship_col][idx]  # Get censorship flag
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', f'{slide_id}')
				features = torch.load(full_path, weights_only=True)
				#return features, label
				return features, event_time, censorship
			
			else:
				#return slide_id, label
				return features, event_time, censorship

		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			#return features, label, coords
			return features, event_time, censorship, coords


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2, event_time_col = 'event_time', censorship_col = 'censorship'):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.event_time_col = event_time_col
		self.censorship_col = censorship_col
		#self.slide_cls_ids = [[] for i in range(self.num_classes)]
		#for i in range(self.num_classes):
		#	self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		


