
config = {'NYT10': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 512,
                'thre_rc' : 0.5, #0.7,
                'thre_ee' : 0.5 #0.8
            },
        'NYT10-HRL': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.5, #0.7,
                'thre_ee' : 0.5 #0.8
            },
        'NYT11-HRL': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.56,
                'thre_ee' : 0.6
            },
        'NYT21-HRL': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.55,
                'thre_ee' : 0.6
            },
        'ske2019': 
            {   
                'plm_name' : 'hfl/chinese-roberta-wwm-ext',
                'maxlen' : 256,
                'thre_rc' : 0.2, # 0.3
                'thre_ee' : 0.1  # 0.2
            },
        'HacRED': 
            {   
                'plm_name' : 'hfl/chinese-roberta-wwm-ext',
                'maxlen' : 502,
                'thre_rc' : 0.7,
                'thre_ee' : 0.4    # 0.7
            },
        'WebNLG': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.2,
                'thre_ee' : 0.1
            },
        'WebNLG_star': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.3,
                'thre_ee' : 0.2
            },
        'CoNLL04': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.5,
                'thre_ee' : 0.5
            },
        'WikiKBP': 
            {   
                'plm_name' : 'bert-base-cased',
                'maxlen' : 256,
                'thre_rc' : 0.5,
                'thre_ee' : 0.5
            }
}