from cortex import Cortex
import time

class Record():
	def __init__(self):
		self.c = Cortex(user, debug_mode=True)
		self.c.do_prepare_steps()

	def create_record_then_export(self,
								record_name,
								record_description,
								record_length_s,
								record_export_folder,
								record_export_data_types,
								record_export_format,
								record_export_version):
		
		self.c.create_record(record_name,
							record_description)

		self.wait(record_length_s)

		self.c.stop_record()

		self.c.disconnect_headset()

		self.c.export_record(record_export_folder,
							record_export_data_types,
							record_export_format,
							record_export_version,
							[self.c.record_id])


	def wait(self, record_length_s):
		print('start recording -------------------------')
		length = 0
		while length < record_length_s:
			print('recording at {0} s'.format(length))
			time.sleep(1)
			length+=1
		print('end recording -------------------------')

# -----------------------------------------------------------
# 
# SETTING
# 	- replace your license, client_id, client_secret to user dic
# 	- specify infor for record and export
# 	- connect your headset with dongle or bluetooth, you should saw headset on EmotivApp
#
# RESULT
# 	- export result should be csv or edf file at location you specified
# 	- in that file will has data you specified like : eeg, motion, performance metric and band power
# 
# -----------------------------------------------------------

user = {
    "license" : "cdb268fe-2039-4fa6-93e5-e778bedac5ae",
    "client_id" : "QMfiqM6tw8CHGLv0aHaCnLv127ID1kJyDzkpnXMr",
    "client_secret" : "T2ebeYpR8JyHSwkCvjE9zJ4uWUZuLvnl1vVv8gFtJPPyQtF9uboosvtFcKIAyia3o1JzZnBzLWKlCJ9jLpClASggfkpcEw6j5ZYj732aoBL4NLPKpCa41Qm6blz3zmGi",
    "debit" : 150
}

r = Record()

# record parameters
record_name = 'oct_28_test'
record_description = 'first test of EEG recording'
record_length_s = 300


# export parameters
record_export_folder = '/Users/lisa/Documents/Gatech/GT Robotics/TAMER/TAMER/EEG-TAMER/recordings'
record_export_data_types = ['EEG', 'MOTION', 'PM', 'BP']
record_export_format = 'CSV'
record_export_version = 'V2'


# start record --> stop record --> disconnect headset --> export record
r.create_record_then_export(record_name,
							record_description,
							record_length_s,
							record_export_folder,
							record_export_data_types,
							record_export_format,
							record_export_version )