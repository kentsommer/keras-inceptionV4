'''
Copyright 2017 Kent Sommer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
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
