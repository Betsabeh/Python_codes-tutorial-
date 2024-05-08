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

