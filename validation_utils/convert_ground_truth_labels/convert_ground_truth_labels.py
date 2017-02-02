classes = eval(open('correct_classes.txt', 'r').read())
correct_classes = {}
old_classes = {}

correct_labels = []

lines = open('old_classes_raw.txt').read().splitlines()

for line in lines:
	_, uid, name = line.split()
	name = name.replace("_", " ")
	name = name.lower()
	uid = int(uid)
	old_classes[uid] = name

for key, value in classes.iteritems():
	uid = key
	name = value.split(",")[0]
	name = name.lower()
	correct_classes[name] = uid

lines = open('val_ground_truth_labels.txt').read().splitlines()

for line in lines:
	key = int(line)
	name = old_classes[key]
	new_label = correct_classes[name]
	print("Old label was: ", key, ". New label is: ", new_label)
	correct_labels.append(new_label)

print("Total labels = ", len(correct_labels))

f=open('../GTL.txt','w')
for label in correct_labels:
	f.write(str(label)+'\n')
f.close()