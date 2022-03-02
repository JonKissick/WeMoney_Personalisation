import unittest
from data_calculation import *
import pandas
# Load pd.DataFrames for test cases

test1 = pd.read_csv('data/posts.csv')
test2 = pd.read_csv('data/users.csv')
test3 = pd.read_csv('data/interest.csv')
test4 = pd.read_csv('data/test_cases/test4.csv')
test5 = pd.read_csv('data/test_cases/test5.csv')
test10 = pd.read_csv('data/test_cases/test10.csv')
test11 = pd.read_csv('data/test_cases/test11.csv')
test12 = pd.read_csv('data/test_cases/test12.csv')

## Test initial data pull from csv

class TestDataGet(unittest.TestCase):
    def test_(self):
        self.assertDictEqual(get_posts().to_dict(),test1.to_dict())
        self.assertDictEqual(get_users().to_dict(), test2.to_dict())
        self.assertDictEqual(get_interests().to_dict(), test3.to_dict())


## Test filtering of users

class TestDataFilter(unittest.TestCase):
    def test_(self):
        self.assertDictEqual(get_user_data('c7a76177-e727-4cf7-afc5-ba79f3d72f20').to_dict(),test4.to_dict())
        self.assertDictEqual(get_user_interest('c7a76177-e727-4cf7-afc5-ba79f3d72f20').to_dict(),test5.to_dict())

# Test outputs of cluster model for a case in each category

class TestClusterModel(unittest.TestCase):
    def test_(self):
        self.assertEqual(cluster_user('542bfa13-54c7-4d79-beb3-b2f797a00c9d'),3)
        self.assertEqual(cluster_user('c7a76177-e727-4cf7-afc5-ba79f3d72f20'),0)
        self.assertEqual(cluster_user('599599cb-43fc-469d-8fed-b9560c48e46f'),2)
        self.assertEqual(cluster_user('8887103e-1841-4cb5-9d31-7d03078dc4cd'),1)


# Test Final sorted frame output

class TestRankedPosts(unittest.TestCase):
    def test_(self):
        self.assertDictEqual(enrich_posts('542bfa13-54c7-4d79-beb3-b2f797a00c9d','2022-02-21').to_dict(),test10.to_dict())
        self.assertDictEqual(enrich_posts('c7a76177-e727-4cf7-afc5-ba79f3d72f20','2022-02-10').to_dict(),test11.to_dict())
        self.assertDictEqual(enrich_posts('599599cb-43fc-469d-8fed-b9560c48e46f','2022-01-26').to_dict(),test12.to_dict())


unittest.main(argv=[''],verbosity=2, exit=False)
