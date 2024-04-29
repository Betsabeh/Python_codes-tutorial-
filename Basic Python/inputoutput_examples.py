import sys

#example of input with map and split
#example 1: 
x, y= input("Enter two values: ").split()
print("First number is {} and second number is {}".format(x, y))

#example 2: 
print("-------example 2--------")
x = list (map(int,input("enter multiple numbers").split()))
print("list values =", x)

#example 3: 
print("-------example 3--------")
name, age = input("Enter your name and age separated by a ,: ").split(',')
print("Hello,", name + "! You are", age, "years old.")

# example 4:  sys.stdin Stdin stands for standard input
#input paramter = size of input character
print("-------example 4--------")
name= sys.stdin.readline(5) 
print(name) 

# example 5 :
print("-------example 5--------")
a=6
b=2
c=2024
print(a,b,c,sep="-")

#example 6: write in file
print('Hello world.!!Hi',file=open('my_file.txt','w'))
