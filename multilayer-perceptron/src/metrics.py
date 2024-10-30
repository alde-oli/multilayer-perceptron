import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Not working yet because python is shit and doesn't let me use references
class Metric:
	# setup the dataframe with column names
	def __init__(self, to_record):
		if not isinstance(to_record, dict):
			raise TypeError("metrics_dict must be a dictionary")
		self.metrics_dict = to_record
		self.data = pd.DataFrame(columns=to_record.keys())
	
	def update(self, live_print=False):
		if not isinstance(live_print, bool):
			raise TypeError("live_print must be a boolean")
		new_line = pd.DataFrame([self.metrics_dict])
		self.data = pd.concat([self.data, new_line], ignore_index=True)
		if live_print:
			print(self.data.loc[self.data.index[-1]])

	def plot(self):
		sns.set_theme()
		self.data.plot()
		plt.show()
# Metric


if __name__ == "__main__":
	loss = 0
	accuracy = 0
	metric = Metric({"loss": loss, "accuracy": accuracy})
	metric.update(live_print=True)
	loss, accuracy = 0.1, 0.5
	metric.update(live_print=True)
	loss, accuracy = 0.2, 0.6
	metric.update(live_print=True)
	metric.plot()
# main
