import unittest
from pathlib import Path

import pandas as pd

from analysis import *


class Test(unittest.TestCase):

    def setUp(self):
        data = pd.read_excel('./data.xlsx', 'data')
        self.dirty = data
        factors = pd.read_excel('./data.xlsx', 'factors')
        # data = data[~data['v20'].isna()]
        data.dropna(inplace=True)
        self.data = data
        self.factors = factors

    def test_all(self):
        self._cronbach_alpha()
        self._factor_scores()
        self._factor_corr()
        self._difficulty()
        self._distinction()
        self._draw_diff_dist()
        self._generate_html()
        main()

    def _cronbach_alpha(self):
        data = self.data
        r= cronbach_alpha(data)
        self.assertEqual(r, 0.7434752166867231)

    def _factor_scores(self):
        data = self.data
        factors = self.factors
        self.fscores = factor_scores(data, factors)

    def _factor_corr(self):
        corr = factor_corr(self.fscores)

    def _difficulty(self):
        diffs = difficulty(self.data[self.factors['item']], self.factors[['item', 'max_score']])
        self.assertEqual(diffs.sum(), 12.638013698630134)
        self.assertEqual(diffs.count(), 20)
        self.diffs = diffs

    def _distinction(self):
        dist = distinction(self.data[self.factors['item']] )
        self.assertEqual(dist.mean(), 0.417608897903321)
        self.assertEqual(dist.count(), 20)
        self.dist = dist

    def _draw_diff_dist(self):
        data = pd.concat([self.diffs, self.dist], axis=1)
        data.columns = ['difficulty', 'distinction']
        draw_diff_dist(data, 'plot.png')

    def _generate_html(self):
        outpath = Path(__file__).parent.absolute()
        generate_report(self.data, self.factors,outpath)

if __name__ == '__main__':
    unittest.main()
