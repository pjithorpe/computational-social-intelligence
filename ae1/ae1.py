import numpy as np
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("data-part-1.csv", header=None, names=["Fnn", "Topic", "Event", "Start time", "End time"])

#clean data

cleaned_data = pd.DataFrame(columns=["Fnn", "Action", "Role", "Gender", "Duration"]);
cleaned_data_list = []
for index, row in data.iterrows():
	# remove any extra events after the first
	event_contents = row['Event'].split(" ")[0]
	# if the event field is 2 strings separated by an underscore
	event_contents = event_contents.split("_")
	if len(event_contents) == 2:
		# and it has a relevant action type
		if event_contents[0] == "laughter" or event_contents[0] == "filler" or event_contents[0] == "bc":
			# and it has both a role code and gender code
			if len(event_contents[1]) == 2:
				# and these codes are valid
				if event_contents[1][0] == "r" or event_contents[1][0] == "c":
					if event_contents[1][1] == "M" or event_contents[1][1] == "F":
						# add this row to the cleaned data
						duration = float(row["End time"]) - float(row["Start time"])
						cleaned_data_list.insert(0, {"Fnn": row["Fnn"], "Action": event_contents[0], "Role": event_contents[1][0], "Gender": event_contents[1][1], "Duration": duration})

cleaned_data = pd.concat([cleaned_data, pd.DataFrame(cleaned_data_list)], ignore_index=True, sort=True)

############################################################

def GetChiSquareResults(obs, exp):
	# Get chi square and p value
	chi_sq_results = stats.chisquare(f_obs=obs, f_exp=exp)

	chi_square = chi_sq_results[0]
	p_value = chi_sq_results[1]

	print("Chi square:", chi_square, "P value:", p_value)


# Part 1 - Test 1
print("\nTest 1")

# Count the number of observed laughs
female_laughs = sum((cleaned_data["Gender"] == "F") & (cleaned_data["Action"] == "laughter"))
male_laughs = sum((cleaned_data["Gender"] == "M") & (cleaned_data["Action"] == "laughter"))
total_laughs = female_laughs + male_laughs

# Calculate total talk time and total female talk time
total_duration_uncleaned = 0.0
female_duration_uncleaned = 0.0
for index, row in data.iterrows():
	event_contents = row['Event'].split(" ")[0].split("_")
	if len(event_contents) > 0 and event_contents[0] != "silence":
		total_duration_uncleaned += float(row["End time"]) - float(row["Start time"])
		if len(event_contents) > 1:
			if len(event_contents[1]) == 2:
				if event_contents[1][1] == "F":
					female_duration_uncleaned += float(row["End time"]) - float(row["Start time"])
			else:
				if event_contents[1] == "F":
					female_duration_uncleaned += float(row["End time"]) - float(row["Start time"])

print("Total duration from all data =", total_duration_uncleaned, "Female duration from all data =", female_duration_uncleaned)

# Calculate expected female laughs and male laughs
all_females = cleaned_data[cleaned_data["Gender"] == "F"]

total_duration = total_duration_uncleaned
female_duration = female_duration_uncleaned
# alternate method - ignores data that isn't laugh, filler or bc
#female_duration = all_females["Duration"].sum()
#total_duration = cleaned_data["Duration"].sum()

fraction_female = female_duration / total_duration
print("p_f =", fraction_female)
print("p_m =", 1 - fraction_female)

expected_female_laughs = fraction_female * total_laughs
expected_male_laughs = total_laughs - expected_female_laughs
print("Observations: Female =", female_laughs, "Male =", male_laughs)
print("Expectations: Female =", expected_female_laughs, "Male =", expected_male_laughs)

GetChiSquareResults([female_laughs, male_laughs], [expected_female_laughs, expected_male_laughs])


# Part 1 - Test 2
print("\nTest 2")

# Count the number of observed laughs
female_fillers = sum((cleaned_data["Gender"] == "F") & (cleaned_data["Action"] == "filler"))
male_fillers = sum((cleaned_data["Gender"] == "M") & (cleaned_data["Action"] == "filler"))
total_fillers = female_fillers + male_fillers

# Calculate expected female laughs and male laughs
expected_female_fillers = fraction_female * total_fillers
expected_male_fillers = total_fillers - expected_female_fillers
print("Observations: Female =", female_fillers, "Male =", male_fillers)
print("Expectations: Female =", expected_female_fillers, "Male =", expected_male_fillers)

GetChiSquareResults([female_fillers, male_fillers], [expected_female_fillers, expected_male_fillers])

############################################################
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

def GetStudentsTResults(vec_x, vec_y, alpha):
	#Calculate the variance with N-1
	variance_x = vec_x.var(ddof=1)
	variance_y = vec_y.var(ddof=1)

	mean_x = vec_x.mean()
	mean_y = vec_y.mean()

	n_x = len(vec_x)
	n_y = len(vec_y)

	print("Female mean =", mean_x)
	print("Male mean =", mean_y)
	print("Female variance =", variance_x)
	print("Male variance =", variance_y)

	# show graphed distributions
	sigma_x = math.sqrt(variance_x)
	sigma_y = math.sqrt(variance_y)
	x = np.linspace(mean_x - 3*sigma_x, mean_x + 3*sigma_x, 100)
	y = np.linspace(mean_y - 3*sigma_y, mean_y + 3*sigma_y, 100)
	plt.plot(x, stats.norm.pdf(x, mean_x, sigma_x), 'r', label='Females')
	plt.plot(y, stats.norm.pdf(y, mean_y, sigma_y), 'b', label='Males')
	plt.xlabel("Duration")
	plt.legend()
	plt.show()


	# Calculate the t-statistic
	t = (mean_x - mean_y) / (np.sqrt( (variance_x / n_x) + (variance_y / n_y) ))

	deg_of_freedom = n_x + n_y - 2

	#p-value after comparison with the t 
	p = 1 - stats.t.cdf(t, df=deg_of_freedom)

	print("My implementation: t =", t)
	print("My implementation: p =", 2*p) # Multiply the p value by 2 because it's a 2 tail t-test

	if p > alpha:
		print('Accept null hypothesis')
	else:
		print('Reject the null hypothesis')

	# Cross Checking with the internal scipy function
	t2, p2 = stats.ttest_ind(vec_x, vec_y, equal_var=False)
	print("SciPy implementation: t = ", t2)
	print("SciPy implementation: p = ", p2)


# Part 1 - Test 3
print("\nTest 3")

all_males = cleaned_data[cleaned_data["Gender"] == "M"]

all_female_laughs = all_females[all_females["Action"] == "laughter"]
all_male_laughs = all_males[all_males["Action"] == "laughter"]

female_laughs_lengths = all_female_laughs["Duration"]
male_laughs_lengths = all_male_laughs["Duration"]

GetStudentsTResults(female_laughs_lengths, male_laughs_lengths, 0.05)


# Part 1 - Test 4
print("\nTest 4")

all_female_fillers = all_females[all_females["Action"] == "filler"]
all_male_fillers = all_males[all_males["Action"] == "filler"]

female_fillers_lengths = all_female_fillers["Duration"]
male_fillers_lengths = all_male_fillers["Duration"]

GetStudentsTResults(female_fillers_lengths, male_fillers_lengths, 0.05)