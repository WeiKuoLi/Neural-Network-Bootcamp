#implement sorting and write unittests

def sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        mid = len(arr)//2
        left = sort(arr[:mid])
        right = sort(arr[mid:])
        return merge(left, right)

def merge(left, right):
    result = []
    while left and right:
        if left[0] < right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left)
    result.extend(right)
    return result

arr = [1,2,3,4,5,6,7,8,9,10]
print(sort(arr))        

#tests
import unittest
class TestSort(unittest.TestCase):
    def test_sort(self):
        arr = [1,2,3,4,5,6,7,8,9,10]
        self.assertEqual(sort(arr), [1,2,3,4,5,6,7,8,9,10])
        
if __name__ == '__main__':
    unittest.main()
