
# Test the testing framework

def import_bo():
    """
    Test if edbo is installed and the main method can
    be instantiated
    """
    
    from edbo.bro import BO
    bo = BO()
    
    return len(bo.obj.domain) == 0

def test_test():
    assert import_bo()