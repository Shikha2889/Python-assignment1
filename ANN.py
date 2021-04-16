#Single Output Vertex:
a = [1.0, 3.5]
c = [3.8, 1.5]
d = -1.7
z = []
result = 0
sum = 0

def linear(w,x,b):
    global result
    for i in range(len(x)):
        z = x[i]*w[i]
        result = result + z

    sum = result + b

    return sum

output = linear(a,c,d)
print("linear")
print(output)

#Output Layer:
x = [1, 3.5]
w = [[3.8, 1.5], [-1.2, 1.1]]
b = [-1.7, 2.5]


def linear_layer(x, w, b):
    result = []
    for i in range(len(b)):
        product = 0
        for j in range(len(x)):
            product = product + w[i][j]*x[j]
        result.append(round(product + b[i], 2))
    return result 

result = linear_layer(x,w,b)
print("linear_layer")
print(result)


#Inner Layers:
x1 = [1,0]
w1 = [[2.1,-3.1], [-0.7, 4.1]]
b1 = [-1.1, 4.2]

def inner_layer(x1,w1,b1):
    for i in range(len(b1)):
        product = 0
        for j in range(len(x1)):
            product = product + w1[i][j]*x1[j]
        result.append(round(product + b1[i], 2))
    return result

result = inner_layer(x1,w1,b1)
print(inner_layer)
print(result)

#Full Inference: 

def inference(x,w,b):
    ans=[]
    n=len(w)
    b1 = b[0]
    x1=x
    for i in range(2):
        inf = linear_layer(x1,w[i],b[i])
        print(inf)
        x1=inf

x=[1,0]
w=[[[2.1,-3.1],[-0.7,4.1]],[[3.8,1.5],[-1.2,1.1]]]
b=[[-1.1,4.2],[-1.7,2.5]]

inference(x,w,b)

#reading Weights: 
def read_weights(file_name):
    wts_matrices = [] # create an empty list to store weight matrices
    f = open(file_name, "r") #open the file
    for x in f:  #iterate over each line in the file 
        m = [] #list to store weight matrix
        if x == "#\n":  #if line contains # then move to the next line
            continue 
        else:
            x = x.split(",") # every element separated by "," will be stored in a lis
        temp = []
        for i in x: # iterate over the new list of elements
            i = float(i) # all the weight are extracted and stored in string format so we have to conver them back to float data type
            temp.append(i) # each element will be stored in the empty list called as temp
            print(temp)
            m.append(temp)
            wts_matrices.append(m)
        return wts_matrices #returns the list of weight matrices

result = read_weights('example_weights.txt')
print("read weight")
print(result)

def read_weights(file_name):
    f = open(file_name,"r")
    l1 = []
    l2 = []
    f.readline()
    for i in f.readlines():
        i = i.rstrip("\n") # removing '\n' char at the end.
        if i == '#':
            l1.append(l2)
            l2 = list()
        else:
            l3 = list(map(float,i.split(',')))
            l2.append(l3)
    l1.append(l2)
    return l1
w = read_weights("weights.txt")
print(w)
print(len(w))
print(len(w[1]))
print(len(w[1][0]))


# read biases 
def read_biases(file_name):
    fl=open(file_name,"r")
    arr = [] #the list arr will store list of bias values
    while True:
        ln = fl.readline() # read the file line by linear_layer
        if not ln: # end of the file has been reached
             break
        if(ln[0]=='#'): # to check if the line starts with #
            continue

        ln = ln.strip() #removes '\n' character
        ls = ln.split(",") # split the values separated by comma and put them in a list
        arr.append(ls) # insetr the list ls in arr
    fl.close()
    return arr

arr = read_biases("example_biases.txt")
print("read biases:")
print(arr)



#read image file
def read_image(filename):
    
  # opening file
  file = open(filename)

  # reading the content of file
  # data will be of type string
  data = file.read()

  # splitting by row, so that each row is a list
  data = data.split("\n")

  # splitting the data in each row
  data = [i.split(" ") for i in data]

  # converting every element of row to int from string
  # now data is a nested list containg data from given file in int type
  data = [[int(j) for j in i] for i in data]

  return data

data = read_image("another_image.txt")

for i in data:
  print(i)


# Output Selection

#function to return first index of maximum element of a list
def argmax(scores):#scores is list of computed scores by the ANN
   
    M = max(scores)#max function returns the maximum value from a list
    print(M)
    
    #index method of a python list returns the first index of passed value if the passed value is in the list
    #as M is maximum value if we pass m in index it will give first index of maximum element
    indx = scores.index(M)
   
    return indx #return the computed index value
    

#main program
print('Enter the scores computed by ANN: ')

#below line of code is used to take comma separated float values as input and store them as a list
x = list(map(float,float(input()).split()))

a = argmax(x)#calling function argmx and storing return value in variable a

print("index of element with max value: ",a)#print the index



#PART 2: 
#Flip a pixel:

def flip_pixel(x):
    if x == 0: 
        x = 1
    else: 
        x = 0
    return x

flip = flip_pixel(0)
print(flip)

#Modify a list:

def modified_list(i, x):
    x[i] = flip_pixel(x[i])
    return x 

x = [1, 0, 1, 1, 0, 0, 0]
i = 2
flip_list = modified_list(i, x)
print(flip_list)

#Quality Computation: 

def compute_difference(x1, x2):
    count = 0 
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            count += 1 
    return count 

x1 = [1, 0, 1, 1, 0, 0, 0]
x2 = [1, 1, 1, 0, 0, 0, 1]

compute = compute_difference(x1, x2)
print(compute)

#Select a pixel

#def select_pixel(x, w, b):
    
# x = read_image(‘image.txt’)
# w = read_weights(‘weights.txt’)
# b = read_biases(‘biases.txt’)
# pixel = select_pixel(x, w, b)

