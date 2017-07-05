#/usr/bin/env python3

from wrenlab.ncbi.geo.label import Evaluation
from wrenlab.ncbi.geo import label
from wrenlab.ontology import fetch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import *
#from toolbox import match_keys
from io import StringIO

from joblib import Memory
memory = Memory(cachedir="cache/")

#sns.set_context('paper', font_scale=4)

@memory.cache
def ontology_distance_actual(return_data=False):
    ev = Evaluation(9606)
    evo = ev.tissue()
    D = evo.distance()
    if return_data:
        return D
    sns.distplot(D.Distance, kde=False)
    plt.ylabel("Count")
    plt.xlim((0,15))


@memory.cache
def ontology_distance_random(return_data=False):
    ev = Evaluation(9606)
    evo = ev.tissue()
    D = evo.distance(randomize=True)
    if return_data:
        return D
    sns.distplot(D.Distance, kde=False)
    plt.ylabel("Count")
    plt.xlim((0,15))


def plot_onto_dist():
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, sharex=False)

    D = ontology_distance_actual(return_data=True)
    p1 = sns.distplot(D.Distance, kde=False, ax=ax1)
    plt.ylabel("Count")
    plt.xlim((0,15))

    D = ontology_distance_random(return_data=True)
    p2 = sns.distplot(D.Distance, kde=False, ax=ax2)
    plt.ylabel("Count")
    plt.xlim((0,15))

    plt.tight_layout()



def tissues_dict_in_platforms(ti):
    BTO = fetch('BTO')
    tissues = []
    for platform in ti['predictions'].keys():
        tissues = tissues + ti['predictions'][platform].columns.tolist()
    tissues = set(tissues)
    return {BTO._resolve_id(int(i)):BTO.name_map()[i]
                for i in tissues}

@memory.cache
def tissues_dict():
    BTO = fetch('BTO')
    return {BTO._resolve_id(int(i)):BTO.name_map()[i]
                for i in BTO.name_map().keys()}

def make_y_gender(ge, ma, platform):
    #for platform in ge['predictions'].keys():
    y_pred = ge['predictions'][platform].applymap( lambda x: int(np.exp(x) > 0.5) )
    #y_pred = ge['predictions'][platform].applymap(np.exp)
    #y_pred = y_pred.apply(lambda x: y_pred.columns[x == 1][0], axis=1)
    #y_pred = pd.DataFrame(y_pred.values, index = ['GSM'+str(i) for i in y_pred.index],
    #                        columns = ['gender'])

    y_pred = y_pred[['M']]
    y_pred.index = ['GSM'+str(i) for i in y_pred.index]

    y_true = ma[['gender']]
    y_true = y_true.dropna()[y_true != 'MIXED']
    y_true = pd.get_dummies(y_true, prefix='', prefix_sep='')[['M']]

    y_pred, y_true = y_pred.align(y_true, axis=0, join='inner')

    return y_true, y_pred



def make_y(ti, ma,  platform, sample_size=None):
    BTO = fetch('BTO')
    tissues_d = tissues_dict(ti)
    tissues = ti['predictions'][platform].columns
    tissues = [BTO._resolve_id(int(i)) for i in tissues]

    y_true = ma[['tissue_id']][[ i in tissues for i in ma.tissue_id ]]

    if sample_size:
        y_pred = ti['predictions'][platform].sample(2000, random_state=3)
    else:
        y_pred = ti['predictions'][platform]
    y_pred = pd.DataFrame(y_pred.values, index = ['GSM'+str(i) for i in y_pred.index],

                          columns = [BTO._resolve_id(int(i)) for i in  y_pred.columns])
    #print(y_pred.columns)

    y_pred = y_pred.apply(lambda x: y_pred.columns[x == x.max()][0], axis=1)
    y_pred = pd.DataFrame(y_pred, columns = ['tissue_id'])
    #assert sum([i in y_true.index for i in y_pred.index]) > 0

    y_pred, y_true = y_pred.align(y_true, axis=0, join='inner')
    return y_true, y_pred


def make_y2(ti, ma,  platform, sample_size=None):
    '''multiclass'''
    BTO = fetch('BTO')
    tissues_d = tissues_dict(ti)
    tissues = ti['predictions'][platform].columns
    tissues = [BTO._resolve_id(int(i)) for i in tissues]

    y_true = ma[['tissue_id']] #[[ i in tissues for i in ma.tissue_id ]]
    y_true = pd.get_dummies(y_true, prefix='', prefix_sep='')

    if sample_size:
        y_pred = ti['predictions'][platform].sample(2000, random_state=3)
    else:
        y_pred = ti['predictions'][platform]

    y_pred = pd.DataFrame(y_pred.values, index = ['GSM'+str(i) for i in y_pred.index],
                          columns = [BTO._resolve_id(int(i)) for i in  y_pred.columns])

    y_pred = y_pred.applymap( lambda x: np.exp(x) > 0.5 )

    #assert sum([i in y_true.index for i in y_pred.index]) > 0

    y_pred, y_true = y_pred.align(y_true, axis=None, join='inner')
    return y_pred, y_true

    #y_pred = y_pred.apply(lambda x: y_pred.columns[x == 1][0] if sum(x==1) ==1 else None, axis=1)

#@memory.cache
def make_y3_(ti, ma):
    '''
    multiclass all platforms
    can't really pool it tho - why did I think that?
    '''
    BTO = fetch('BTO')
    tissues_d = tissues_dict(ti)

    ti2 = {}
    for platform in ti['predictions'].keys():
        x = ti['predictions'][platform]
        x = x.sample(100, random_state=3)
        x = x.applymap( np.exp )
        x = pd.DataFrame(x.values,
                    index = ['GSM'+str(i) for i in x.index],
                    columns = [BTO._resolve_id(int(i)) for i in x.columns])
        #x.reset_index(inplace=True)
        #x.rename(columns = {'index':'sid'}, inplace=True)
        #x = x.apply(lambda x_: x.columns[x_ == 1][0] if sum(x_==1) ==1 else None, axis=1)
        ti2[platform] = x

    return ti2

'''
def make_y_merged(ti, ma):
    ti2 = make_y3_(ti, ma)

    all_tissues = []
    for i in ti2.values():
        all_tissues = all_tissues+i.columns.tolist()
    all_tissues = list(set(all_tissues))

    mean_proba = {}
    for t in all_tissues:
        tmp = {}
        for platform in ti2.keys():
            if t in ti2[platform].columns:
                tmp[platform] = ti2[platform][t]
        return tmp
            mean_proba[t] =
'''

def make_y3_obsol(ti2, ti, ma):
    y_pred1 = pd.merge(ti2[570], ti2[96], how='outer')#, right_index=True, left_index=True, on=[])
    y_pred2 = pd.merge(ti2[6244], ti2[6947], how='outer')#, right_index=True, left_index=True)
    y_pred = pd.merge(y_pred1, y_pred2, how='outer')#, right_index=True, left_index=True)
    y_pred.index = y_pred.sid
    y_pred.drop('sid', axis=1, inplace=True)

    y_true = ma[['tissue_id']] #[[ i in tissues for i in ma.tissue_id ]]
    #y_true = pd.get_dummies(y_true, prefix='', prefix_sep='')
    #y_true = pd.DataFrame(y_true.values,
    #            index = ['GSM'+str(i) for i in y_true.index],
    #            #columns = [BTO._resolve_id(int(i)) for i in  y_true.columns]
    #            )
    #y_pred = y_pred.apply(lambda x: y_pred.columns[x == 1] if x.max == 1 else None, axis=1)
    #y_true = y_true.apply(lambda x: y_true.columns[x == 1][0], axis=1)
    y_pred.dropna(inplace=True)

    y_pred, y_true = y_pred.align(y_true, axis=None, join='inner')
    return y_true, y_pred

from collections import defaultdict

def reverse_dummy(df_dummies):
    pos = defaultdict(list)
    vals = defaultdict(list)

    for i, c in enumerate(df_dummies.columns):
        if "_" in c:
            k, v = c.split("_", 1)
            pos[k].append(i)
            vals[k].append(v)
        else:
            pos["_"].append(i)

    df = pd.DataFrame({k: pd.Categorical.from_codes(
                        np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                        vals[k])
                    for k in vals})

    df[df_dummies.columns[pos["_"]]] = df_dummies.iloc[:, pos["_"]]
    return df


def confmat(y_true, y_pred, ti, x='all'):
    tissues_d = tissues_dict(ti)
    if x != 'all':
        tissues_d = {k:v for k,v in tissues_d.items() if k in x}
    print(tissues_d)
    cm = pd.DataFrame(
                confusion_matrix(y_true, y_pred, labels = [i for i in tissues_d.keys()]),
                            columns = [i for i in tissues_d.values()],
                            index = [i for i in tissues_d.values()]
                )
    return cm



def inspection_spreadsheet():
    from wrenlab.ncbi.geo import label
    from wrenlab.ontology import fetch
    BTO = fetch('BTO')
    text = label.sample_text()

    lge = label.extract_all_gender(9606)
    lag = label.extract_all_age(9606)
    lti = label.extract_all_tissue(9606)
    lti = lti.apply(lambda x: BTO._resolve_id(int(x)) if pd.notnull(x) else None)

    o = text
    o, lti = text.align(lti, axis=0, join='left')
    o, lge = text.align(lge, axis=0, join='left')
    o, lag = text.align(lag, axis=0, join='left')

    o['nlp_gender'] = lge
    o['nlp_age'] = lag
    o['nlp_tissue_id'] = lti

    #add manual
    ma = pd.read_csv('geo_manual_labels_jdw.tsv', sep='\t', index_col=0)
    ma.index = [int(i.strip('GSM')) for i in  ma.index]
    ma.index.name = 'SampleID'
    o, ma = o.align(ma, axis=0, join='left')

    o['gold_gender'] = ma.gender
    o['gold_age'] = ma.age
    o['gold_tissue_id'] = ma.tissue_id

    # insert tissue names
    tn1 = o['nlp_tissue_id']
    tn2 = o['gold_tissue_id']

    #tidi = BTO.name_map()
    tidi = {BTO._resolve_id(int(k)):v for k,v in BTO.name_map().items()}
    tn1 = tn1.apply(lambda z: tidi[z] if pd.notnull(z) else None)
    tn2 = tn2.apply(lambda z: tidi[z] if pd.notnull(z) else None)

    o['nlp_tissue_name'] = tn1
    o['gold_tissue_name'] = tn2

    o = o[[0,1,2,3,4,5,6,8,9,10,14, 11,12,13,15]]

    # OPTIONAL remove those with no predictions nor gold
    ss = o[['gold_gender', 'gold_age', 'gold_tissue_id', 'gold_tissue_name',  'nlp_gender',
            'nlp_age', 'nlp_tissue_id', 'nlp_tissue_name']]\
            .dropna(0, 'all').index

    ss2 = o[['gold_gender', 'gold_age', 'gold_tissue_id', 'gold_tissue_name']]\
            .dropna(0, 'all').index
    ss3 = o[['nlp_gender', 'nlp_age', 'nlp_tissue_id', 'nlp_tissue_name']]\
            .dropna(0, 'all').index
    x = pd.Index(set(ss2).intersection(set(ss3)))

    o = o.ix[ss]


    return o


def get_metrics_tissue(ti, ma, metr):
    ''' metr    an empty or populated dict'''
    for platform in ti['predictions'].keys():
        yt, yp = make_y2(ti, ma, platform)
        if yp.shape[0] > 0:
            metr['tissue_precision_macro_'+str(platform)] = precision_score(yt, yp, average='macro')
            metr['tissue_recall_macro_'+str(platform)] = recall_score(yt, yp, average='macro')
            metr['tissue_precision_micro_'+str(platform)] = precision_score(yt, yp, average='micro')
            metr['tissue_recall_micro_'+str(platform)] = recall_score(yt, yp, average='micro')

    return metr



def get_metrics_gender(ge, ma, metr):
    ''' metr    an empty or populated dict'''
    for platform in ge['predictions'].keys():
        y_true, y_pred = make_y_gender(ge,ma, platform)

        metr['gender_precision_'+str(platform)] = precision_score(y_true, y_pred)
        metr['gender_recall_'+str(platform)] = recall_score(y_true, y_pred)
    return metr

'''
def metrics_table(metr):
    ms = pd.DataFrame(
                    columns=['Gender', 'Tissue (macro)', 'Tissue (micro)'],
                    index= ['Precision', 'Recall']
                    )
    for c in ms.columns:
        for i in ms.index:
            c_ = [s_.strip('(').strip(')').lower() for s_ in c.split(' ')]
            i_ = [s_.strip('(').strip(')').lower() for s_ in i.split(' ')]
            val = np.mean(list(match_keys(metr, c_+i_).values()))
            ms.ix[i,c] = val

    return ms



    metr_summ = pd.Series({
        'gender_recall': np.mean(list(match_keys(metr, ['gender', 'recall']).values())),
        'gender_precision': np.mean(list(match_keys(metr, ['gender', 'precision']).values())),
        'tissue_recall': np.mean(list(match_keys(metr, ['tissue', 'recall']).values())),
        'tissue_precision': np.mean(list(match_keys(metr, ['tissue', 'precision']).values())),
        })
'''

def pool_predictions_tissue(ti, ma):
    # make a predictions table with collapsed probabilities from ti['predictions'] tables
    # if many - use mean
    pass


def tp(y_true, y_pred):
    yt = pd.get_dummies(y_true, prefix="", prefix_sep="")
    yp = pd.get_dummies(y_pred, prefix="", prefix_sep="")
    yt, yp = yt.align(yp, join='inner')
    return y_true == y_pred

def tp_macro():
    return tp(y_true, y_pred).sum().sum()

def tn(y_true, y_pred, colname):
    return sum(y_true[colname] == y_pred[colname])

#accuracy_score(y_true, y_pred) # (TP+TN)/(P+N) = (TP+TN)/all
#recall_score(y_true, y_pred) # TPR
#precision_score(y_true, y_pred) # TP/(TN+FN)
#f1_score # 2 * prec * sens / (prec+sens) (harmonic mean)



def cr_pandas(y_true, y_pred):
    y_pred, y_true = y_pred.align(y_true, axis=0, join='inner')

    cr = classification_report(y_true, y_pred)
    cr = pd.read_fwf(StringIO(cr))
    cr.set_index('Unnamed: 0', inplace=True)
    cr.dropna(inplace=True)
    cr.index.name = "tissue_id"
#cr.drop('avg / total', inplace=True)
#cr = cr.ix[cr.support > use_min_support]
    #cr = cr.drop(set(cr.index).difference(use_tissues +['avg / total']))
    return cr

