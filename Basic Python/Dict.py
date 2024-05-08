#Example of nested dictionary
Dic= {"name":"John", "age":30, "job":"teache", "address":{"street":"str","build":123,"city":"Shz"}}
print("len of Dic", len(Dic))
print("Dic values:", Dic.values())
print("Dic keys:", Dic.keys())
print("Address street=:", Dic["address"]["street"])

