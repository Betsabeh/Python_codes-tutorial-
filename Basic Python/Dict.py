# 1-Example of nested dictionary
Dic= {"name":"John", "age":30, "job":"teacher", "address":{"street":"str","build":123,"city":"Shz"}}
print("len of Dic", len(Dic))
print("Dic values:", Dic.values())
print("Dic keys:", Dic.keys())
print("Address street=:", Dic["address"]["street"])

# 2-using for for accessing the dictionary values
for key in Dic.keys():
  print("key:",key, " value=", Dic[key])

# 3- Another way of printing Dic
print([(k, Dic[k]) for k in Dic])

# 4- change values in Dic
Dic['name'] = 'Mary'

# 5-convert to list
list_Dic = list (Dic.items())
print(list_Dic)
print(list_Dic[2]) # print list index=2
Dic = dic(list_Dic) #convert back to the Dictionary

# 6-Merge 2 Dic
def Merge_Dic(D1,D2):
  merged_dict = D1.copy()
  return (merged_dict.update(D2))

D1={"a":2,"B":3}
D2={"g":5, "3":4 , "r":5}
Merge_Dic(D1,D2)
print(D1)  



