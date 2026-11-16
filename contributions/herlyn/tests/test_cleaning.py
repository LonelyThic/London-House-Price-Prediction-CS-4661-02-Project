"""Run: python -m unittest discover -s contributions/herlyn/tests -v."""
import unittest
import numpy as np
import pandas as pd
from contributions.herlyn.cleaning import PropertyCleaner


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            'ID':[1,2,3], 'price':[100.,200.,300.],
            'outcode':['A','A','B'], 'tenure':['Freehold','Freehold','Shared'],
            'propertyType':['Terraced','Detached House',None],
            'currentEnergyRating':['c',None,'None'],
            'bedrooms':[2.,4.,np.nan], 'bathrooms':[1.,2.,np.nan],
            'livingRooms':[1.,1.,np.nan], 'floorAreaSqM':[80.,120.,np.nan]})

    def test_training_only_imputation_and_raw_preservation(self):
        original = self.train.copy(deep=True)
        cleaner = PropertyCleaner(impute=True).fit(self.train)
        valid = self.train.iloc[[2]].copy()
        valid.index = [99]
        out = cleaner.transform(valid)
        self.assertEqual(out.loc[99,'bedrooms_clean'],3.)
        self.assertEqual(out.loc[99,'bedrooms_missing'],1)
        self.assertEqual(out.loc[99,'currentEnergyRating_clean'],'Missing')
        pd.testing.assert_frame_equal(self.train,original)
        pd.testing.assert_frame_equal(out[valid.columns],valid)
        changed = valid.copy(); changed['bedrooms'] = 900.
        cleaner.transform(changed)
        self.assertEqual(cleaner.medians_['bedrooms'],3.)

    def test_categories_and_unseen_tenure(self):
        cleaner = PropertyCleaner(group_rare=True, rare_fraction=.5).fit(self.train)
        valid = self.train.copy();valid.loc[0,'tenure']='New tenure'
        out = cleaner.transform(valid)
        self.assertEqual(out.loc[0,'tenure_clean'],'Other')
        self.assertEqual(out.loc[2,'tenure_clean'],'Other')
        self.assertEqual(out.loc[0,'propertyType_clean'],'Terrace Property')
        self.assertEqual(out.loc[1,'propertyType_clean'],'Detached House')
        self.assertEqual(out.loc[0,'currentEnergyRating_clean'],'C')

    def test_invalid_numerics_and_nan_native_mode(self):
        cleaner = PropertyCleaner().fit(self.train)
        valid = self.train.copy();valid.loc[0,'floorAreaSqM']=-1
        valid.loc[1,'bathrooms']=np.inf
        out = cleaner.transform(valid)
        self.assertTrue(pd.isna(out.loc[0,'floorAreaSqM_clean']))
        self.assertEqual(out.loc[0,'floorAreaSqM_missing'],1)
        self.assertTrue(pd.isna(out.loc[1,'bathrooms_clean']))


if __name__ == '__main__':
    unittest.main()
