
#### Normalization values for SEVIR dataset ####
# Values taken from SEVIR repo: https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/src/readers/normalizations.py
PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
    'ir069': 1 / 1174.68,
    'ir107': 1 / 2562.43,
    'vil': 1 / 47.54,
    'lght': 1 / 0.60517}
PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
    'ir069': 3683.58,
    'ir107': 1552.80,
    'vil': - 33.44,
    'lght': - 0.02990}

#### Selected events from test dataset for visualization ####
bestScores = ['S855443','S853391','S853298','S855170','R19092007067565','R19092606047771',
              'S853188','R19081806217856','R19062600507914','S851388','S851130','S856018',
              'R19091920207746','R19092606047806','S851839','R19083001247951','S857576',
              'S852075','S834575','S857334']

worstScores = ['R19082620197760','R19082617508061','R19103019347835','R19061613148219',
               'R19081316428478','R19102822407767','R19112205207295','S845788',
               'R19112310377604','R19103005428034','R19092705428309','R19082917097578',
               'R19080121067959','R19092410447877','R19112416258199','R19112421488415',
               'R19111204337541','R19112016107635','R19081805108540','R19082712337424']

randScores = ['R19070915538410', 'S846087', 'R19083001247563', 'R19081608527826',
            'R19100622287527', 'S829040', 'R19082920497394', 'R19091021348435',
            'R19111819348279', 'S844398', 'R19111813348432', 'R19111123427753',
            'R19092511058374', 'R19112016107658', 'R19072715377452', 'R19102617517408',
            'R19110912357245', 'R19080809007611', 'S853622', 'R19100900257102']