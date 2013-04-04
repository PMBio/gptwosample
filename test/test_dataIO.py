'''
Created on Apr 3, 2013

@author: Max
'''
import unittest
from gptwosample.data.dataIO import get_data_from_csv
from StringIO import StringIO
import numpy

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testRead(self):
        d = get_data_from_csv('test.csv')
        for name in ['g1','g2','othername']:
            assert name in d
            assert d[name].shape == (2,4)
        assert numpy.allclose(d["othername"][0],[1,2,3,4])
        assert numpy.all(d['input'][numpy.isfinite(d["input"])] == [1.,2.,3.])
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()