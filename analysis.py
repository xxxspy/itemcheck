import pandas as pd 
import numpy as np
from pathlib import Path
from decimal import Decimal
import webbrowser

SUM_NAME = 'max_score'

def trueround_precision(number, places=0, rounding=None)->Decimal:
    '''
    trueround_precision(number, places, rounding=ROUND_HALF_UP)

    Uses true precision for floating numbers using the 'decimal' module in
    python and assumes the module has already been imported before calling
    this function. The return object is of type Decimal.

    All rounding options are available from the decimal module including 
    ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_DOWN, ROUND_HALF_EVEN, 
    ROUND_HALF_UP, ROUND_UP, and ROUND_05UP.

    examples:

        >>> trueround(2.5, 0) == Decimal('3')
        True
        >>> trueround(2.5, 0, ROUND_DOWN) == Decimal('2')
        True

    number is a floating point number or a string type containing a number on 
        on which to be acted.

    places is the number of decimal places to round to with '0' as the default.

    Note:   if type float is passed as the first argument to the function, it
            will first be converted to a str type for correct rounding.

    GPL 2.0
    copywrite by Narnie Harshoe <signupnarnie@gmail.com>
    '''
    from decimal import ROUND_HALF_UP
    from decimal import ROUND_CEILING
    from decimal import ROUND_DOWN
    from decimal import ROUND_FLOOR
    from decimal import ROUND_HALF_DOWN
    from decimal import ROUND_HALF_EVEN
    from decimal import ROUND_UP
    from decimal import ROUND_05UP

    if type(number) == type(float()):
        number = str(number)
    if rounding == None:
        rounding = ROUND_HALF_UP
    place = '1.'
    for i in range(places):
        place = ''.join([place, '0'])
    return Decimal(number).quantize(Decimal(place), rounding=rounding)

def cronbach_alpha(items: pd.DataFrame):
    '''Cronbach’s alpha信度系数
    items: 题目数据, 每列是一个题目'''
    items_count = items.shape[1]
    variance_sum = items.var(axis=0, ddof=1).sum()
    total_var = float(items.sum(axis=1).var(ddof=1))
    return (items_count / float(items_count - 1) *
            (1 - variance_sum / total_var))


def cronbach_alpha_std(items: pd.DataFrame):
    '''Cronbach’s alpha信度系数
    items: 题目数据, 每列是一个题目'''
    items_count = items.shape[1]
    corr = items.corr('pearson')
    index = np.triu_indices(items_count)
    corr.values[index] = 0
    s = corr.sum().sum()
    n = items_count * (items_count-1)/2
    mean = s / n
    return items_count * mean / (1+(items_count-1)*mean)


def factor_scores(items: pd.DataFrame, factors: pd.DataFrame):
    '''计算平均数'''
    fs = []
    for factor, group in factors.groupby('factor'):
        items[factor] = items[group['item'].values].mean(axis=1)
        fs.append(factor)
    items['sum_factors'] = items[fs].sum(axis=1)
    return items[fs+['sum_factors']]

def factor_corr(fscores):
    return fscores.corr()

def difficulty(items: pd.DataFrame, max_scores: pd.DataFrame):
    diff = pd.Series([None]*items.shape[1], index=max_scores['item'])
    max_scores = max_scores.set_index('item')[SUM_NAME]
    max_scores = max_scores.to_dict()
    for i in max_scores:
        n = items[i].count()
        s = items[i].sum()
        diff.loc[i] = s / (n*max_scores[i])
    return diff


def distinction(items: pd.DataFrame):
    dist = pd.Series([None]*items.shape[1], index=items.columns)
    total = items.sum(axis=1)
    for i in items.columns:
        item = items[i]
        dist.loc[i]=np.corrcoef(total.values, item.values)[0, 1]

    return dist

def draw_diff_dist(data: pd.DataFrame, filename):
    print(data)
    ax = data.plot()
    ax.set_xticks(range(data.shape[0]))
    ax.set_xticklabels(list(data.index), ha="center", rotation = 90)
    ax.get_figure().savefig(filename)

def diff_table(diff: pd.Series):
    diff = diff.map(lambda x: trueround_precision(x, 3))
    head = '<tr><td>难度</td><td>难度描述</td><td>题目数量</td></tr>'
    rtn = [head,]
    groups = [
            (0, 0.199), 
            (0.2, 0.399),
            (0.4, 0.699),
            (0.7, 0.799),
            (0.8, 1)
        ]
    labels = ['难','较难','中等','较易','容易',]

    i = 0
    for g in groups:
        label = labels[i]
        n = sum((diff >= g[0]) &( diff <=g[1]))
        i += 1
        row = f'<tr><td>{g[0]}~{g[1]}</td><td>{label}</td><td>{n}</td></tr>'
        rtn.append(row)
    discribe = {}
    discribe['最大难度值'] = diff.max()
    discribe['最小难度值'] = diff.min()
    discribe['平均难度值'] = diff.mean()
    for k,v in discribe.items():
        row = f'<tr><td>{k}</td><td>{v}</td></tr>'
        rtn.append(row)
    rows = '\n'.join(rtn)
    return f'<table class="table table-striped">{rows}</table>'

def dist_table(diff: pd.Series):
    diff = diff.map(lambda x: trueround_precision(x, 3))
    head = '<tr><td>区分度</td><td>区分度描述</td><td>题目数量</td></tr>'
    rtn = [head,]
    groups = [
            (0, 0.199), 
            (0.2, 0.299),
            (0.3, 0.399),
            (0.4, 1)
        ]
    labels = ['需要修改','修改之后会更好','合格','较好',]

    i = 0
    for g in groups:
        label = labels[i]
        n = sum((diff >= g[0]) &( diff <=g[1]))
        i += 1
        row = f'<tr><td>{g[0]}~{g[1]}</td><td>{label}</td><td>{n}</td></tr>'
        rtn.append(row)
    discribe = {}
    discribe['最大区分度值'] = diff.max()
    discribe['最小区分度值'] = diff.min()
    discribe['平均区分度值'] = diff.mean()
    for k,v in discribe.items():
        row = f'<tr><td>{k}</td><td>{v}</td></tr>'
        rtn.append(row)
    rows = '\n'.join(rtn)
    return f'<table class="table table-striped">{rows}</table>'

def item_quality(data: pd.DataFrame, diff_dist: pd.DataFrame):
    diff_dist.columns = ['难度', '区分度']
    diff_dist['删除该题后的试卷信度'] = None
    for i in data.columns:
        cols = [c for c in data.columns if c != i]
        subdf = data[cols]
        r = cronbach_alpha(subdf)
        diff_dist.loc[i, '删除该题后的试卷信度']=r
    for c in diff_dist.columns:
        diff_dist[c] = diff_dist[c].map(lambda x: trueround_precision(x, 3))
    return diff_dist


def generate_report(data: pd.DataFrame, factors: pd.DataFrame, outpath: Path):
    template = Path(__file__).parent.absolute() / 'template.html'
    template = template.read_text(encoding='utf8')
    reliability=cronbach_alpha(data)
    fscores = factor_scores(data, factors)
    pearson = factor_corr(fscores).to_html()
    pearson = pearson.replace('class="dataframe"', 'class="table table-striped"')
    img_path = outpath / 'diff-dist.png'
    diff = difficulty(data[factors['item']], factors[['item', 'max_score']])
    dist = distinction(data[factors['item']])
    diff_dist =  pd.concat([diff, dist], axis=1)
    diff_dist.columns = ['difficulty', 'distinction']
    draw_diff_dist(diff_dist, str(img_path))
    diff_dis_img = f'<img src="file://{img_path}" />'

    diff_table_ = diff_table(diff)
    dist_table_ = dist_table(dist)
    item_quality_ = item_quality(data[factors['item']], diff_dist).to_html()
    item_quality_ = item_quality_.replace('class="dataframe"', 'class="table table-striped"')
    html = template.format(
            reliability=reliability, 
            pearson=pearson,
            diff_dis_img=diff_dis_img,
            diff_table = diff_table_,
            dist_table = dist_table_,
            item_quality = item_quality_
        )
    report_path = outpath / 'report.html'
    Path(report_path).write_text(html, encoding='utf8')


def read_excel(fpath):
    data = pd.read_excel('./data.xlsx', 'data')
    factors = pd.read_excel('./data.xlsx', 'factors')
    data.dropna(inplace=True)
    return data, factors


def main():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    fpath = filedialog.askopenfilename()
    data, factors = read_excel(fpath)
    outpath = Path(fpath).parent
    generate_report(data, factors, outpath)
    report_path = outpath / 'report.html'
    webbrowser.open(report_path)

main()