print('This is Ser-501 --Assignment 2 --Problem 1 --Part A')


from numpy import asarray
import numpy as np

ENERGY_LEVEL = [100, 113, 110, 85, 105, 102, 86, 63,
                81, 101, 94, 106, 101, 79, 94, 90, 97]

# ==============================================================


# The brute force method to solve first problem
def find_significant_energy_increase_brute(A):

    """
    Return a tuple (i,j) where A[i:j] is the most significant energy increase
    period.
    we have to find the Max sum of subarray 
    1. First Compute the Difference from the current - previous 
    2. Find the Min and Max
    2. FInd the maximum the sum(of differences) of subarray 
    3. Return the Array  
    time complexity = O(n^2)
    """
    
    
    n = len(A)
    start_index = 0
    end_index = 0
    max_increase = float('-inf')

    for i in range(n):
        current_increase = 0
        for j in range(i+1, n):
            current_increase += A[j] - A[j-1]  #Index 1 (minus) Index 2  + Adding up the differences
            if current_increase > max_increase :
                max_increase = current_increase
                start_index = i
                end_index = j
    
    return start_index, end_index
                                
# ==============================================================

   
    #for the sub-array sum 
    
def find_significant_energy_increase_recursive(A, low, high):
    
         # Case 1, Array Size = 1
    
    if low==high :  
        return low, high, 0
    
    mid = (low + high) // 2
    
    #Left Best Sub Array 
    left_low, left_high , left_sum = find_significant_energy_increase_recursive(A, low, mid)

    #Right Best Sub Array 
    right_low, right_high , right_sum = find_significant_energy_increase_recursive(A, mid+1, high)
    
    # Best Sub Array crossing the Middle (mid) 
    cross_low, cross_high , cross_sum = max_crossing(A, low, mid, high)
    
    
    #return the Subarray. Left ?? Right ?? Crossing
    
    if left_sum >= right_sum and left_sum >= cross_sum : 
        return left_low, left_high, left_sum
    
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return right_low , right_high, right_sum
    
    else:
        return cross_low, cross_high, cross_sum
    
# ==============================================================


# The recursive method to solve first problem
def max_crossing(A, low, mid, high):

    """
    Return a tuple (i,j) where A[i:j] is the most significant energy increase
    period.
    time complexity = O (n logn)
    
    1. Divide the Array in two parts from the Middle
    2. Now calculate the Sum towards left and Right of Sub-Array
    3. Find the best Subarray, Left(of Middle) , Crossing(through the middle) , Right(of middle)
    [100, 113, 110, 85, 105, 102, 86, 63, //81\\, 101, 94, 106, 101, 79, 94, 90, 97]
    """
    

    #For Left Side Calculation    
    left_sum = float('-inf') #left side sum as Negative Infinity
    total_sum = 0
    max_left = mid

    #traverse from Mid to Left of Array and calcuate the Difference
    for i in range (mid, low - 1 , -1):  # (first, last, jump of)
        if i > low :
            total_sum += ( A[i] - A[i-1] )
        else :
            total_sum += 0
            
        if total_sum > left_sum : 
            left_sum = total_sum
            max_left = i
            
             
    #For Right Side Calculation
    right_sum = float('-inf') #left side sum as Negative Infinity
    total_sum = 0 
    max_right = mid + 1 #Center to right first element in array
    
    for j in range(mid+1 , high+1): 
        total_sum = total_sum + ( A[j] - A[j-1] )  #calculate difference
        if total_sum > right_sum :   #update the crrent total sum in array traverse right
            right_sum = total_sum
            max_right = j
        
    cross_sum = left_sum + right_sum    
    return max_left, max_right, cross_sum
    

# The iterative method to solve first problem
def find_significant_energy_increase_iterative(A,start_index, end_index):

    """
    Return a tuple (i,j) where A[i:j] is the most significant energy increase
    period.
    time complexity = O(n)
    1. Traverse through array, get the Sum
    2. if sum == -ve , reset to 0 , updates the Indexes
    3. Track max increase and update   
    """
    
    
    n = len(A)
    start_index = 0  #best subarray starting index
    end_index = 0       #best subarray end index
    temp_start = 0      #store new temp potential start index if new max found
    max_increase = float('-inf')
    current_increase = 0
    
    for i in range(1,  n):
        current_increase = current_increase + A[i] - A[i-1] #differncce between thr Consective elements
        
        if current_increase > max_increase : 
            max_increase = current_increase
            start_index = temp_start
            end_index = i
            
            
        if current_increase < 0 :  # if sum negative start over "Discard the subaray"
            current_increase = 0
            temp_start = i  #now where the Index got the Zero start from there



    return start_index, end_index
# ==============================================================    


# The Strassen Algorithm to do the matrix multiplication

'''
What will we do in this questions,
1. Add or Subtract matrix , Merge and split matrix for 7 multiplications
2. 
'''

def add_matrix(A,B):
    return [[A[i][j] + B[i][j]
    for j in range(len(A))]
    for i in range(len(A))]
    
def sub_matrix(A,B):
    return [[A[i][j] - B[i][j]
    for j in range(len(A))]
    for i in range(len(A))]
    
def identity_matrix(matrix):
    size = len(matrix)
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    
def split_matrix(A):
    n = len(A)
    mid = n // 2
    return ( 
      [row[:mid] for row in A[:mid]],
      [row[mid:] for row in A[:mid]],
      [row[:mid] for row in A[mid:]],
      [row[mid:] for row in A[mid:]]

    )
    
def merge_matrix(C11, C12, C21, C22):
    n = len(C11)
    C = [[0 for _ in range(2*n)] for _ in range(2*n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = C11[i][j]
            C[i][j+n] = C12[i][j]
            C[i+n][j] = C21[i][j]
            C[i+n][j+n] = C22[i][j]
    return C



def square_matrix_multiply_strassens(A, B):

    """
    Return the product AB of matrix multiplication.
    Assume len(A) is a power of 2
    """

    A = asarray(A)
    n = len(A)
    B = asarray(B)

    assert A.shape == B.shape
    assert A.shape == A.T.shape
    assert (len(A) & (len(A) - 1)) == 0, "A is not a power of 2"

    
    
    if len(A) == 1 : 
        return [[A[0][0] * B[0][0]]]
    
    mid = n // 2
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    #Split operation on the Matrix 
    A11 , A12 , A21 , A22  = split_matrix(A)
    B11 , B12 , B21 , B22  = split_matrix(B)
    
    # calculate 7(seven) products 
    
    P1 = square_matrix_multiply_strassens(add_matrix(A11, A22), add_matrix
    (B11, B22))
    P2 = square_matrix_multiply_strassens(add_matrix
    (A21, A22), B11)
    P3 = square_matrix_multiply_strassens(A11, sub_matrix
    (B12, B22))
    P4 = square_matrix_multiply_strassens(A22, sub_matrix
    (B21, B11))
    P5 = square_matrix_multiply_strassens(add_matrix
    (A11, A12), B22)
    P6 = square_matrix_multiply_strassens(sub_matrix
    (A21, A11), add_matrix
    (B11, B12))
    P7 = square_matrix_multiply_strassens(sub_matrix
    (A12, A22), add_matrix(B21, B22))


    #Combine the product get he resukts
    
    C11 = add_matrix(sub_matrix(add_matrix(P1, P4), P5), P7)
    C12 = add_matrix(P3, P5)
    C21 = add_matrix(P2, P4)
    C22 = add_matrix(sub_matrix(add_matrix(P1, P3), P2), P6)    
    return merge_matrix(C11, C12, C21, C22)

# ==============================================================


# Calculate the power of a matrix in O(k)
def power_of_matrix_naive(A, k):
    """
    Return A^k.
    time complexity = O(k)
    """
    
    result = A
    for _ in range(k-1):
        result = square_matrix_multiply_strassens(result, A)
        
    return result

# ==============================================================


# Calculate the power of a matrix in O(log k)
def power_of_matrix_divide_and_conquer(A, k):
    """
    Return A^k.
    k is even >> A^k/2
    k is odd >> A^k-1 and multiply by A
    time complexity = O(log k)
    """
    
    #CHECK IF NOT TOUCHED (//CHATGPT)
    if not isinstance(k, int):
        raise TypeError(f"Expected integer for exponent, got {type(k).__name__}")
    
    
    if k == 0:
        return identity_matrix(A)
    
    if k == 1 : 
        return A
    
    half_power = power_of_matrix_divide_and_conquer(A, k // 2)

    if k%2 == 0:
        return square_matrix_multiply_strassens(half_power, half_power)
    
    else : 
        return square_matrix_multiply_strassens(square_matrix_multiply_strassens(half_power,half_power),A )





# ==============================================================


def test1():
     
    assert find_significant_energy_increase_brute(ENERGY_LEVEL) == (7, 11)
    print("Testing brute force method... PASSED")

    recursive_result = find_significant_energy_increase_recursive(ENERGY_LEVEL, 0, len(ENERGY_LEVEL) - 1)
   # print(f"Recursive result: {recursive_result}") 
    assert recursive_result[:2] == (7,11)
    print("Testing Recursive method... PASSED")

    assert find_significant_energy_increase_iterative(ENERGY_LEVEL, 0 , len(ENERGY_LEVEL)-1) == (7, 11)
    print("Testing Iterative method... PASSED")


def test2():
    A = [[0, 1], [1, 1]]
    #B = A
    k = 3 
    
    assert np.array_equal(square_matrix_multiply_strassens(A,A), np.dot(A,A))
    print("Testing Strassen Operation.... PASSED")
    assert np.array_equal(power_of_matrix_naive(A,k), np.linalg.matrix_power(A,k))
    print("Testing Matrix Naive.... PASSED")
    assert np.array_equal(power_of_matrix_divide_and_conquer(A,k),np.linalg.matrix_power(A,k))
    print("Testing Matrix Divide and Conqueor.... PASSED")


def test_recursive_energy_increase():
    case = [
        (ENERGY_LEVEL,0,len(ENERGY_LEVEL) - 1 , (8,11)),   #Expectrd Results
        ([10,20,30,40,50], 0, 4, (0,4)),
        ([50,40,30,20,10], 0, 4, (0,0)),
        ([5,-2,3,8,-4, 6,-1,10],0,7,(2,7)),
        ([10,5,10,5,10],0,4,(0,2)),
        
    ]
    
    for A,low,high,expected in case:
        result = find_significant_energy_increase_recursive(A,low,high)[:2]
        assert result == expected, f"Failed {A}: Expected: {expected}, Output: {result}"
    print("All Recursive Test Passed")
    
    
def test_strassen_matrix_multiplication():
    case = [
        ( [[0]],[[0]],[[0]] ),
        ( [[1]],[[1]],[[1]] ),
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
         ([[2, 3], [4, 5]], [[1, 0], [0, 1]], [[2, 3], [4, 5]]), 
          ([[0, 1], [1, 1]], [[1, 1], [1, 0]], [[1, 0], [2, 1]]),
          ([[5, -2], [-3, 7]], [[2, 3], [4, -1]], [[2, 21], [16, -14]]),
    ]
    
    for A, B, expected in case: 
        assert np.array_equal(square_matrix_multiply_strassens(A,B), expected), \
            f"Failed for {A} * {B} == {expected}"
            
    print("Strassen Matrix Multiplication::")
            
            
        
def test_matrix_power() : 
    A = [[2,3], [1,4]]
    identity = identity_matrix(A)
    case = [                        #check for the powers 
        (A,0,identity),
        (A,1,A),
        (A,2,np.dot(A,A)),
        (A, 3, np.dot(np.dot(A, A), A)),
        ([[-1,2],[3,-4]], 2, np.dot([[-1,2],[3,-4]], [[-1,2],[3,4]])),
    
    ]
    
    for matrix, k , expected in case: 
        assert np.array_equal(power_of_matrix_naive(matrix,k), expected), \
            f"Naive Failed {matrix}^{k}"
        assert np.array_equal(power_of_matrix_divide_and_conquer(matrix, k), expected),\
            f"Divide and Conqueor Failed : {matrix}^{k}"
    
    print('All MATRIX PASSED Naive &  Divide and Conqueorr')
    
    
def test_significant_energy_level_increase(): 
    case = [
        ([100, 113, 110, 85, 105, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97], 7, 11),  
        ([10], 0, 0),  
        ([10, 10, 10, 10], 0, 0),  
        ([50, 40, 30, 20, 10], 0, 0), 
        ([10, 20, 30, 40, 50], 0, 4), 
        ([5, -2, 3, 8, -4, 6, -1, 10], 2, 7),  
        ([10, 5, 10, 5, 10], 0, 2),  
        ([1, -1, 2, -2, 3, -3, 4, -4], 6, 6),  
        ([0, 0, 0, 1], 2, 3),  
    ]
     
    for A, expected_start, expected_end in case : 
        assert find_significant_energy_increase_brute(A) == (expected_start, expected_end), \
            f"Brute Force Failed {A}"

        recursive_results =  find_significant_energy_increase_iterative(A,0,len(A)-1) 
        assert recursive_results[:2] == (expected_start, expected_end), \
            f"Recursive Failed for {A}, Expected : {(expected_start, expected_end)}, Output as : {recursive_results[:2]}"


        assert find_significant_energy_increase_iterative(A) == (expected_start, expected_end), \
            f"Iterative fails for {A}"


    print("All Passes, Brute , Recursive, Iterative")
        
        
        
        

if __name__ == '__main__':
    # test1()
    # test2()
    test_recursive_energy_increase()
    test_strassen_matrix_multiplication()
    test_matrix_power()
    test_significant_energy_level_increase()