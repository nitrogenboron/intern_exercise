from util import *


def test_functions():
    # Test is_concave
    f_concave = np.array([1, 2, 3, 2, 1])
    assert is_concave(f_concave) == True, "Should be True"

    f_concave = np.array([1, 2, 2, 2, 3])
    assert is_concave(f_concave) == False, "Should be False"

    # Test is_convex
    f_convex = np.array([1, 2, 3, 2, 1])
    assert is_convex(f_convex) == False, "Should be False"

    f_convex = np.array([1, 2, 2, 2, 3])
    assert is_convex(f_convex) == False, "Should be False"
    
    f_convex = np.array([2, 2, 2, 2, 3])
    assert is_convex(f_convex) == True, "Should be True"

    # Test is_non_decreasing
    f_non_decreasing = np.array([1, 2, 3, 3, 4])
    assert is_non_decreasing(f_non_decreasing) == True, "Should be True"

    f_non_decreasing = np.array([1, 2, 3, 2, 4])
    assert is_non_decreasing(f_non_decreasing) == False,"Should be False"

    # Test is_non_increasing
    f_non_increasing = np.array([4, 3, 3, 2, 1])
    assert is_non_increasing(f_non_increasing) == True, "Should be True"

    f_non_increasing = np.array([4, 3, 3, 4, 1])
    assert is_non_increasing(f_non_increasing) == False,"Should be False"


    # Test valid_u
    valid_u_values = np.array([10, 20, 25,65],np.int32)
    assert valid_u(valid_u_values) == True, "Should be True"

    invalid_u_values = np.array([10, 20, 25,65,30])
    assert valid_u(invalid_u_values) == False,"Should be False"

    
    #test function validation
    a,b,c,d=0,10,1,10
    u=np.array([u for u in range(1, 300) if u % 5 == 0 and u % 3 != 0])
    assert validate(example_f1,a,b,c,d,u)==True,  "Should be True"  
    

    assert validate(example_f2,a,b,c,d,u)==True,  "Should be True" 

    #test search_maxima        
    history=search_maxima( bbo_func=lambda x: -(x-150)**2) 
    assert history[-1]==150, "Should be 150"
    
    history=search_maxima( bbo_func=lambda x: np.sinc(x/10)) 
    assert history[-1]==0, "Should be 0"

    history=search_maxima( bbo_func=lambda x: -0.1*(x-30)**2) 
    assert history[-1]==30, "Should be 30"
        
    print("Util.py: All tests passed!")
    
if __name__ == "__main__":
    test_functions()

