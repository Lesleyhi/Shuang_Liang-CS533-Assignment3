test_sizes = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

Mflops_julia = \
[[0.001859375,0.00265625,0.0036875,0.00490625,0.0065,0.007875,0.009625,0.0118125,0.014125,0.017,0.01975,0.023125,0.02725],[0.00184375,0.00265625,0.00371875,0.00484375,0.006375,0.0076875,0.0095625,0.0116875,0.014375,0.01675,0.019625,0.023125,0.02675],[0.00184375,0.00265625,0.00371875,0.00484375,0.00625,0.00775,0.0099375,0.01175,0.014125,0.01675,0.019625,0.023125,0.02675],[0.00184375,0.00265625,0.00371875,0.0048125,0.0063125,0.00775,0.009625,0.011625,0.026,0.016875,0.019625,0.02325,0.0395],[0.00184375,0.002625,0.00371875,0.00484375,0.0064375,0.00775,0.0095625,0.0116875,0.01425,0.017125,0.0535,0.023,0.0275],[0.001875,0.00265625,0.00371875,0.004875,0.0063125,0.00775,0.0096875,0.011625,0.0145,0.01675,0.019625,0.023,0.03975],[0.001859375,0.00265625,0.00371875,0.00484375,0.00634375,0.00775,0.009625,0.011625,0.014125,0.017125,0.01975,0.023,0.02675],[0.001859375,0.002671875,0.00371875,0.00490625,0.006375,0.00775,0.0096875,0.0116875,0.014375,0.016875,0.01975,0.022875,0.03975],[0.001859375,0.0026875,0.00371875,0.00484375,0.006375,0.00775,0.009625,0.011625,0.01425,0.017125,0.019625,0.02325,0.02675],[0.001875,0.00265625,0.0036875,0.00484375,0.0063125,0.0076875,0.0095625,0.011625,0.014375,0.01675,0.019625,0.02325,0.02675],[0.001859375,0.00265625,0.0036875,0.004875,0.0063125,0.00775,0.009625,0.011625,0.014375,0.01675,0.01975,0.022875,0.03975],[0.001875,0.002671875,0.00375,0.004875,0.006375,0.00775,0.009625,0.011625,0.01425,0.017125,0.019625,0.023125,0.02675],[0.001875,0.002859375,0.00371875,0.004875,0.0063125,0.00775,0.009625,0.0116875,0.02525,0.01675,0.01975,0.022875,0.0395],[0.001859375,0.00265625,0.00371875,0.00484375,0.006375,0.00775,0.0095625,0.011625,0.014375,0.017,0.019625,0.02325,0.02675],[0.00190625,0.00265625,0.0036875,0.00484375,0.006375,0.00775,0.0095625,0.011625,0.0145,0.01675,0.019625,0.023,0.02675],[0.001859375,0.00271875,0.0036875,0.00484375,0.00625,0.0076875,0.0095625,0.0116875,0.01425,0.01675,0.019625,0.02325,0.0265],[0.001875,0.002671875,0.00371875,0.00503125,0.006375,0.00775,0.0095625,0.0116875,0.0145,0.01675,0.0195,0.022875,0.02675],[0.001890625,0.00265625,0.0036875,0.00484375,0.00625,0.0076875,0.0095625,0.0116875,0.01425,0.01675,0.01975,0.02325,0.0265],[0.001875,0.00265625,0.00371875,0.00484375,0.00621875,0.00775,0.009625,0.011625,0.014375,0.01675,0.019625,0.023,0.0265],[0.00184375,0.002640625,0.00371875,0.00484375,0.00625,0.00775,0.009625,0.0115625,0.014625,0.016875,0.019625,0.023125,0.02675]]

Mflops_genvect = \
[[0.0160341,0.0233189,0.0311715,0.0417525,0.0542486,0.0703701,0.087296,0.106655,0.128327,0.155413,0.188719,0.221704,0.253097],[0.0160345,0.0233185,0.0311728,0.0417683,0.0542389,0.070352,0.0873724,0.106664,0.128278,0.155401,0.188663,0.221633,0.253215],[0.0160321,0.0233167,0.031169,0.0417643,0.0542364,0.070365,0.087307,0.106674,0.128292,0.155404,0.188633,0.221718,0.253189],[0.0160329,0.0233199,0.0311738,0.0417542,0.0542325,0.0703781,0.0872899,0.106657,0.128322,0.155375,0.188703,0.221739,0.253149],[0.0160335,0.0233179,0.0311723,0.041773,0.0542405,0.0703505,0.0873085,0.106677,0.128288,0.155377,0.188871,0.221675,0.253127],[0.0160309,0.0233165,0.0311715,0.0417445,0.0542345,0.070374, 0.08731,0.106669,0.128272,0.155369,0.188676,0.221619,0.253233],[0.0160314,0.023321,0.0311685,0.0417473,0.0542295,0.0703684,0.087308,0.106677,0.128279,0.155395, 0.18873,0.221672,0.253119],[0.0160331,0.0233204,0.031165,0.0417405,0.054229,0.0703595,0.087324,0.106664, 0.12827,0.155383,0.188711,0.221656,0.253197],[0.0160328,0.0233202,0.0312043,0.0417475,0.0542356,0.0703635,0.087352,0.106712,0.128316,0.155386,0.188806,0.221807,0.253217],[0.0160324,0.0233174,0.031174,0.041749,0.0542295,0.070361,0.0873021,0.106687,0.128284,0.155406,0.188694,0.221641,0.253192],[0.0160337,0.0233165,0.031172,0.0417663,0.0542345,0.070367,0.0872885,0.106658,0.128295,0.155372,0.188684,0.221719,0.253192],[0.0160311,0.023317,0.0311682,0.0417473,0.0542409,0.0703775,0.0873045,0.106685,0.128269,0.155388,0.188667,0.221679,0.253161],[0.0160321,0.0233224,0.0311695,0.041765,0.0542315,0.0703754,0.0873305,0.106683,0.128268,0.155414,0.188732,0.221634,0.253199],[0.0160313,0.0233233,0.0311728,0.0417425,0.0542381,0.070362,0.0872965,0.106624,0.128268,0.155424,0.188999,0.221713,0.253179],[0.0160301,0.0233164,0.0311653,0.0417425,0.0542395,0.0703834,0.0873024,0.106641,0.128297, 0.15538,0.188676,0.221603,0.253165],[0.0160332,0.0233714,0.0311717,0.0417632,0.0542334,0.0703585,0.0872985,0.106659,0.128282,0.155395,0.188675,0.221719,0.253165],[0.0160339,0.0233151,0.0311758,0.0417578,0.054233,0.0703626,0.087304,0.106668, 0.12828,0.155391,0.188648,0.221619,0.253082],[0.0160329,0.0233181,0.0311707,0.0417593,0.054234,0.070367,0.0872999,0.106653,0.128319,0.155429,0.188663,0.221652,0.253144],[0.0160337,0.0233175,0.0311673,0.0417492,0.0542284,0.070369,0.0872935,0.106699,0.128298,0.155397,0.188682,0.221647,0.253117],[0.0160329, 0.02332, 0.03117,0.0417572,0.0542485,0.0703685,0.0872995, 0.10665,0.128282,0.155376,0.188669,0.221692,0.253128]]

Mflops_autovect = \
[[0.0273247,0.0394695,0.0542825,0.0721309,0.0924345,0.119427, 0.14794,0.181848,0.218496,0.264014,0.314661,0.369998,0.431489],[0.0273188,0.0394681,0.0542901,0.0720445,0.0924335,0.118899,0.147977,0.181902,0.218585,0.264004,0.314541,0.370108,0.431531],[0.0273283,0.039471, 0.05431,0.0721165,0.0924031,0.118921,0.147964, 0.18191,0.218519,0.264149,0.314505,0.370242,0.431477],[0.027328,0.039483,0.0543015,0.0720965,0.0924239,0.118875,0.147966,0.181864,0.218523, 0.26402,0.314538, 0.37076,0.431502],[0.0273297,0.0394698,0.0542824,0.0720716,0.092427,0.118892,0.147963,0.181886,0.218511,0.264031,0.314541,0.370145,0.431543],[0.027324,0.039469,0.0542874,0.0720795,0.092411,0.118912,0.147962,0.181913,0.218563,0.264134,0.314531,0.370062,0.431581],[0.0273318,0.039475,0.0543095,0.0720991,0.0924081,0.118942,0.147949,0.181869,0.218488,0.264188,0.314511,0.370075,0.431502],[0.0273253,0.0394733,0.0542854,0.0720755,0.092417,0.118907,0.147945,0.181898, 0.21856,0.264036,0.314552,0.370077,0.431454],[0.027331,0.0394795,0.0543056,0.072134,0.0924045,0.118899,0.147982,0.181863,0.218488,0.264048,0.314528,0.370021,0.431493],[0.0273305,0.0394785,0.0542945,0.0721195,0.0924495,0.118898,0.147954, 0.18204,0.218505,0.264131,0.314585,0.370049,0.431501],[0.0273318,0.0394762,0.054294,0.0720985,0.0924159,0.118879,0.147968, 0.18189,0.218494,0.264037,0.314539,0.370088,0.431482],[0.0273285,0.039476,0.0542915,0.0721135,0.0927035,0.118892,0.147949,0.181911,0.218555,0.264038,0.314517,0.369993,0.431514],[0.0273257,0.0394788,0.054285,0.0720925,0.092433,0.118898,0.147962,0.181866,0.218536,0.264038, 0.31451,0.370056,0.431471],[0.0273305,0.039469,0.0542794,0.072083,  0.0924,0.118899,0.147957,0.181929,0.218501,0.264051, 0.31454,0.370061,0.431517],[0.0273167,0.0394865,0.0542775,0.0721109,0.092422,0.118884,0.147918,0.181896,0.218504,0.264083,0.314553,0.370013,0.431558],[0.0273255,0.0394782,0.054293,0.072113,0.092433, 0.11889,0.147975,0.181951,0.218521,0.263983,0.314536,0.370076,0.431459],[0.0273315,0.039472,0.054286,0.0720714,0.092453,0.118897,0.147952,0.181922, 0.21852,0.264064,0.314545,0.370027,0.431522],[0.0273358,0.0394717,0.0542845,0.0721035,0.0924209,0.118947,0.147956,0.181907,0.218494,0.264047, 0.31459,0.370081,0.431463],[0.027334,0.0394772,0.0542835,0.072119,0.0924075,0.118904,0.147988, 0.18193,0.218528,0.264045, 0.31467,0.370063,0.431469],[0.0273235,0.039475,0.0542926,0.072116,0.0924156, 0.11893,0.147975,0.181902,0.218494,0.264055,0.314591,0.369979,0.431488]]

Mflops_basic = \
[[0.0975925,0.110526,0.181235,0.266163,0.349861,0.437044,0.548923,0.673651,0.766816,0.976533, 1.16483, 1.39032, 1.79819],[0.0975829,0.109613,0.179484,0.266136,0.349625,0.436966,0.548884,0.674005,0.766907,0.976283, 1.16489, 1.39042, 1.79838],[0.0976645,0.109952,0.179491,0.266205, 0.34958,0.437003,0.549243,0.673609,0.766826,0.976252, 1.16488, 1.39017, 1.79729],[0.0976036,0.109619,0.179518,0.266151,0.349659, 0.43703,0.548894,0.673672,0.766856,0.976226, 1.16471, 1.39046, 1.79756],[0.0975955,0.109635,0.179537, 0.26616,0.349593,0.437062,0.548992,0.673894, 0.76686,0.976275, 1.16477, 1.39037, 1.79748],[0.097587,0.109605,0.179504,0.266173,0.349595,0.436997,0.548886,0.673648,0.766835,0.976235, 1.16469, 1.39028, 1.79738],[0.097605,0.109625,0.179509,0.266168,0.349575,0.437005,0.548996, 0.67371,0.766943,0.976422, 1.16486, 1.39014, 1.79986],[0.0975984,0.109602,0.179506,0.266197,0.349615,0.437025,0.548866,0.674397,0.766856,0.976214, 1.16485, 1.39019, 1.79719],[0.0975901, 0.10978,0.179488,0.266192,0.349512,0.436997,0.548894, 0.67366,0.766828,0.976238, 1.16475, 1.39023, 1.79713],[0.0975885,0.109592,0.179641,0.266172,0.349572,0.437001,0.548924,0.673749,0.766899,0.976256, 1.16482,  1.3908, 1.79785],[0.0975934,0.109613,0.179511,0.266158,0.349532,0.437019,0.548931,0.673653,0.766823,0.976206, 1.16481, 1.39016, 1.79775],[0.0976065,0.109591, 0.17993,0.266136,0.349605,0.437011,0.548887,0.673691,0.766879,0.976249, 1.16479, 1.39028, 1.79834],[0.0975975,0.109908, 0.17952,0.266164,0.349592,0.437011,0.548894,0.673628,0.766854,0.976199, 1.16484, 1.39011, 1.79725],[0.0976084,0.109609,0.179549,0.266175,0.349586,0.437302,0.548936,0.673695,0.766849,0.976209, 1.16477, 1.39031, 1.79743],[0.0976011,0.109625,0.179542,0.266178,0.349618,0.437026,0.548963,0.673658, 0.76684,0.976253, 1.16474, 1.39004, 1.79734],[0.0975935,0.109606,0.179713,0.266146,0.349651,0.437041,0.548901, 0.67365,0.766839,0.976203, 1.16469, 1.39027, 1.79743],[0.0975879,0.109605,0.179527,0.266164,0.349553,0.437019,0.549004,0.673665, 0.76693,0.976417, 1.16483,  1.3902, 1.79852],[0.0976444,0.109792,0.179991,0.266161,0.349601,0.437019,0.548893, 0.67366,0.766858, 0.97621, 1.16494, 1.39012, 1.79704],[0.0975986,0.109617,0.179499,0.266143,0.349554,0.437046,0.548926,0.673672,0.766835,0.976262, 1.16477, 1.39021, 1.79715],[0.097595,0.109595,0.179518,0.266137,0.349594,0.437049, 0.54893,0.673731,0.766867,0.976242, 1.16483, 1.39024,  1.7978]]

Mflops_blocked = \
[[0.0302225,0.0433175,0.059581,0.0791689,0.102313, 0.13065, 0.16435,0.199992,0.240052,0.291815,0.350601,0.412066,0.476531],[0.0302253,0.0433203,0.059797,0.079157,0.102327,0.130654,0.164346,0.200017,0.240032,0.291861,0.350657,0.412072,0.476482],[0.0302205,0.0433205,0.0595651,0.0791501,0.102322,0.130663,0.164359,0.199985, 0.24008,0.291856,0.350616,0.412057,0.476486],[0.0302358,0.0433213,0.0595635,0.0791585,0.102336,0.130656, 0.16441,0.199987,0.240066,0.291855,0.350658,0.412029,0.476499],[0.0302248,0.0433215,0.059574,0.0791601, 0.10232,0.130669,0.164399,0.199991,0.240149,0.291854,0.350626,0.412093,0.476505],[0.0302335,0.0433213,0.059587,0.079161,0.102323,0.130657,0.164362,0.200007,0.240071,0.291851, 0.35059,0.412023,0.476482],[0.0302345,0.043318,0.0595665,0.079147,0.102771,0.130652,0.164363,0.200034,0.240087,0.291838,0.350602,0.412043,0.476524],[0.030241,0.0433202,0.0595605,0.079167,0.102326,0.130688, 0.16437, 0.19999,0.240052,0.291862,0.350712,0.411971,0.476503],[0.0302313,0.0433335,0.0595745,0.079152,0.102339,0.130664,0.164395,0.199994,  0.2401,0.291848,0.350644,0.411998, 0.47649],[0.030238,0.0433198,0.0595645,0.0791485,0.102319, 0.13067,0.164366,0.200017,0.240064,0.291878,0.350688, 0.41199,0.476518],[0.0302365,0.0433192,0.059594,0.0791585,0.102324,0.130648,0.164378,     0.2,0.240107,0.291858, 0.35065,0.412053, 0.47655],[0.0302367,0.0433315, 0.05957,0.079162,0.102348,0.130677,0.164367,0.200135,0.240044,0.291841,0.350657,0.411972, 0.47648],[0.0302283,0.0433157,0.059574,0.0791845,0.102332,0.130664, 0.16436,0.199991,0.240079,0.291852,0.350665,0.411997,0.476501],[0.0302337,0.043322,0.0595624,0.0791531,0.102324, 0.13069,0.164437,0.199988,0.240105, 0.29186,0.350667, 0.41203,0.476574],[0.0302228,0.043318, 0.05957,0.0791445,0.102325,0.130706, 0.16438,0.199988,0.240039,0.291836,0.350608,0.412005,0.476493],[0.0302195,0.0433242,0.0595605,0.0791665,0.102315,0.130634, 0.16435,0.199971,0.240094,0.291853,0.350603,0.412021,0.476524],[0.030222,0.0433192,0.0595645,0.0791715,0.102337,0.130652,0.164391,0.199992,0.240096,0.291845,0.350727, 0.41198,0.476537],[0.0302317,0.0433198,0.059561,0.0791509,0.102321,0.130686,0.164358,0.200033,0.240094,0.291859, 0.35064,0.411963,0.476504],[0.0302233,0.043335,0.059561,0.0791426,0.102353,0.130695,0.164385,0.199988,0.240219,0.291852,0.350621,0.412004,0.476515],[0.0302195,0.0433245,0.0595626,0.0791609,0.102339,0.130717,0.164367,0.199982,0.240102,0.292003,0.350663,0.411961,0.476488]]

Mflops_col = \
[[0.200795,0.285696,0.393563,0.520446,0.686055,0.851344,  1.0654, 1.32203,  1.7535, 1.89443,  2.2435, 2.79556, 3.34932],[0.200794,0.285739,0.393536, 0.52055,0.685243,0.851313, 1.06469, 1.32203, 1.75297, 1.89446, 2.24343, 2.79541,  3.3508],[0.200813, 0.28574,0.393554,0.520446,0.685367,0.851329, 1.06471, 1.32214,   1.753, 1.89437, 2.24333, 2.79547, 3.34934],[0.200828,0.285777,0.393494,0.520383,0.685308,0.851289, 1.06473,   1.322, 1.75302, 1.89458, 2.24383, 2.79539, 3.34906],[0.200789,0.285705,0.393506,0.520455,0.685265,  0.8513, 1.06463, 1.32206, 1.75299, 1.89457, 2.24365, 2.79561, 3.34929],[0.200785,0.285697,0.393485,0.520422,0.685209,0.851325, 1.06468, 1.32198, 1.75288, 1.89446, 2.24351, 2.79539, 3.34949],[0.200856,0.285724,0.393498,0.520386,0.685293,0.851356, 1.06475, 1.32201,  1.7529, 1.89453, 2.24345, 2.79694, 3.34938],[0.200813,0.285678,0.393493,0.520388,0.685279,0.851334, 1.06474, 1.32205, 1.75303, 1.89453, 2.24345, 2.79519, 3.34918],[0.200824, 0.28568,0.393501,0.520527,0.685292,0.851257, 1.06459, 1.32207, 1.75293, 1.89677, 2.24351, 2.79585, 3.34916],[0.200798,0.285683,0.393483,0.520398,0.685242,0.851302, 1.06464,   1.322, 1.75282,  1.8943, 2.24337, 2.79581, 3.34948],[0.200826,0.285715,0.393709,0.520396, 0.68523,0.851265, 1.06536, 1.32193, 1.75292, 1.89446, 2.24346, 2.79532, 3.34939],[0.200808,0.285695,0.393598,0.520472,0.685256,0.851397, 1.06467, 1.32205, 1.75306,  1.8945, 2.24357, 2.79531, 3.35058],[0.200799,0.285692,0.393491,0.520432,0.685243,0.851281, 1.06463, 1.32205, 1.75305, 1.89452, 2.24355,  2.7954, 3.34898],[0.200792, 0.28568,0.393505,0.520362,0.685356,0.851271, 1.06466, 1.32199, 1.75293, 1.89447, 2.24348, 2.79538, 3.34931],[0.200813,0.285755,0.393474,0.520401,0.685267,0.851239, 1.06463, 1.32194, 1.75298, 1.89446, 2.24364, 2.79515, 3.34957],[0.200814,0.285742,0.393535,0.520452,0.685277,0.851389, 1.06463,   1.322, 1.75352, 1.89446,  2.2435, 2.79519, 3.34922],[0.200817, 0.28568,0.393493,0.520395,0.685289,0.851395, 1.06476,   1.322, 1.75292, 1.89462, 2.24349, 2.79542, 3.34917],[0.200824,0.285699,0.393499,0.520386,0.686022,0.851274, 1.06463, 1.32195, 1.75289, 1.89444, 2.24336, 2.79531, 3.34899],[0.200799,0.285689,0.393488,0.520398,0.685209,0.851305, 1.06462, 1.32207, 1.75298, 1.89449, 2.24354,  2.7951, 3.35005],[0.200806,0.285694,0.393513,0.520713,0.685211,0.851368, 1.06463,   1.322, 1.75296, 1.89444, 2.24355, 2.79525, 3.35024]]

Mflops_copy = \
[[0.0299595,0.0427895,0.0587389,0.0783759,0.101995,0.129592,0.161254, 0.19858,0.240482,0.288976,0.344325,0.404097,0.473491],[0.029958,0.0427992,0.058805,0.0783989,0.102078,0.129612,0.161288,0.198566,0.240481,0.288938,0.344272,0.404156,0.473587],[0.0299625,0.0429012,0.0587355,0.078377,0.101919,0.129571,0.161269,0.198551,0.240481,0.288816,0.344287,0.404062,0.473582],[0.0299593,0.0427945,0.058735,0.0783805,0.102213, 0.12959,0.161264,0.198531,0.240503,0.288922,0.344869, 0.40406,0.473584],[0.0299617, 0.04279,0.0587405,0.0784615,0.102213,0.129598,0.161269,0.198565,0.240489,0.288553,0.344281,0.404058,0.473514],[0.0299638,0.0427908,0.0587366,0.0783811, 0.10197,0.129593,0.161249,0.198578,0.240506,0.288951, 0.34333, 0.40404,0.473609],[0.0299645,0.042783,0.058743,0.0783801,0.102224, 0.12958,0.161265,0.198535, 0.24049,0.288927,0.344308,0.404083,0.473577],[ 0.02996,0.0427925,0.0587515,0.078729,0.101944,0.129616,0.161245,0.198554,0.240496,0.288561, 0.34427,0.404095,0.473439],[0.0299597,0.0427873,0.0587615,0.0783916,0.102219,0.129587,0.161257,0.198543,0.240518,0.288954,  0.3443,0.404122,0.473573],[0.029965,0.042791,0.0587415,0.078389,0.102203,0.129577,0.161255,0.198567,0.240504,0.288607,0.344298,0.404024,0.473601],[0.029964,0.0427955,0.058748,0.078383,0.102211,0.129595,0.161259,0.198563,0.240506,0.288973,0.344282,0.404534,0.473346],[0.0299635,0.042784,0.058743,0.0783825,0.102204,0.129565,0.161254,0.198589, 0.24048,0.288938,0.344257, 0.40502,0.473534],[0.029964,0.042779,0.058741,0.078384,0.102144,0.129603,0.161253,0.198566,0.240483,0.288923,0.344272,0.404022,0.473587],[0.0299608,0.0427815,0.0587316,0.0783775,0.102215,0.129594,0.161261,0.198559,0.240499,0.288943,0.344097,0.404064,0.473592],[0.0299643,0.042785,0.058735,0.078388,0.102011,0.129588,0.161278, 0.19856,0.240521,0.288927,0.344306,0.404081,0.473482],[0.029969,0.0427895,0.058761,0.0783825,  0.1022,0.129566, 0.16125,0.198537,0.240503,0.288935,0.344252,0.404133,0.473562],[0.0299648,0.0427822,0.0587435,0.0789125,0.102229,0.129575,0.161242, 0.19856,0.240477, 0.28893, 0.34428,0.404086,0.473618],[0.0299645,0.042801,0.0587395,0.0783886,0.102207,0.129586, 0.16126, 0.19855,0.240481,0.288949,0.344287,0.404096,0.473574],[0.0299625,0.0427855,0.058741,0.0783935,0.102206,0.129568,0.161242,0.198562,0.240489,0.288933,0.344255,0.404086, 0.47353],[0.0299705,0.0427848,0.058755,0.0784185,0.102217,0.129603,0.161274,0.198558,0.240481,0.288878,0.344302,0.404109,0.473557]]

Mflops_naive = \
[[0.103753,0.119531,0.187309,0.280971,0.364885,  0.4632,0.578867,0.715293,0.817044, 1.03583, 1.22956, 1.47491,  1.9511],[0.103658,0.119527,0.187243,0.280853,0.364894,0.463239,0.578813,0.715302,0.817062, 1.03574, 1.22968, 1.47271, 1.95016],[0.103642,0.119512, 0.18726,0.280829,0.364903,0.463263, 0.57875,0.715218,0.817121, 1.03577, 1.22972,   1.473,  1.9496],[0.103665,0.119521, 0.18724,0.280859,0.364872,0.463479,0.578784,0.715257,0.817087, 1.03581, 1.22961, 1.47255, 1.94952],[0.103663,0.119514,0.187226, 0.28084,0.364856,0.463255, 0.57881,0.715248,0.817083, 1.03593, 1.22971, 1.47299, 1.94988],[0.103647,0.119506,0.187258,0.280845,0.364839,0.463276,0.578747,0.715274, 0.81716,  1.0358, 1.22968, 1.47263, 1.94974],[0.103654,0.119512, 0.18725,0.280831,0.364853, 0.46324,0.578864,0.715242,0.817081, 1.03581, 1.22994, 1.47245, 1.94961],[0.103651,0.119499,0.187239,0.280918,0.364847, 0.46321,0.578755,0.715284,0.817099, 1.03583, 1.22967, 1.47251, 1.95176],[0.103657,0.119516,0.187259,0.280845,0.364859,0.463256,0.578725,0.715225,0.817006, 1.03576, 1.22973, 1.47367, 1.94978],[0.103642,0.119511,0.187241,0.280803,0.364857,0.463239,0.578748,0.715214,0.817153, 1.03584, 1.22973, 1.47342, 1.95005],[0.103655, 0.11954,0.187294,0.280861,0.364886,0.463239,0.578764,0.715211,   0.817, 1.03587, 1.22968, 1.47266, 1.94985],[0.103668,0.119516,0.187246,0.280836,0.364891,0.463259,0.578769,0.715232,0.817061, 1.03577, 1.22957, 1.47289, 1.95038],[0.103653,0.119509,0.187247,0.280831, 0.36486,0.463285,0.578786,0.715231,0.817178, 1.03574, 1.22969, 1.47546, 1.95031],[ 0.10371,0.119524,0.187262,0.280856,0.364878,0.463239,0.578812,0.715228,0.817032, 1.03571, 1.22964, 1.47274, 1.94993],[0.103643, 0.11952,0.187251,0.280902, 0.36487,0.463224,0.578822,0.715283,0.817165, 1.03581, 1.22967, 1.47245, 1.95003],[0.103639,0.119517,0.187256,0.280844,0.364917,0.463312,0.578777,0.715245,0.817099, 1.03574, 1.22956, 1.47299, 1.95017],[0.103664,0.119526,0.187272,0.280885,0.364852,0.463224,0.578836,0.715226,0.817074, 1.03581,  1.2297, 1.47259,    1.95],[0.103641,0.119526,0.187217, 0.28085,0.364889,0.463271,0.578823,0.715238,0.817141, 1.03581,  1.2296, 1.47238, 1.95417],[0.103646,0.119528,0.187254,0.280803,0.364941,0.463244,0.578747,0.715258,0.817066, 1.03576, 1.22955, 1.47286, 1.95197],[0.103632,0.119528, 0.18724,0.280833,0.364857, 0.46324,0.578803,0.715276,0.817298, 1.03582, 1.22972, 1.47289, 1.95274]]

Mflops_rb = \
[[0.0299155,0.042795,0.0586905,0.0784845,0.102166,0.130611,0.162296,0.200114,0.241321,0.291247,0.343598,0.407259,0.476043],[0.0299158,0.042796,0.0586956,0.0784839,0.102181,0.130585,0.162319,0.200111,0.241307, 0.29122,0.344074,0.407221,0.476072],[0.0299125,0.0427902,0.0586835,0.0785115,0.101949,0.130556,0.162306,0.200112,0.241309,0.291246, 0.34396,0.407192,0.476033],[0.0299125,0.0427938,0.0587076,0.078511,0.102124,0.130562,0.162301,0.200118,0.241268,0.291228,0.344152,0.407203,0.476048],[0.029915,0.0427835,0.0586865,0.0784755,0.102164,0.130584,0.162322,0.200102,0.241456,0.291278,0.344122,0.407285,0.476228],[0.0299225,0.0428087,0.058704,0.0784975,0.101982,0.130573, 0.16236,0.200139,0.241301,0.291256,0.343942, 0.40725,0.476054],[0.0299132,0.0427918,0.0586904,0.078494,0.102156, 0.13094,0.162306,0.200113,0.241328,0.291229, 0.34409,0.407514,0.476044],[0.0299107,0.0428012,0.0587015,0.078509,0.102197,0.130577,0.162312,0.200102,0.241318,0.291215,0.343949,0.407213, 0.47605],[0.0299103,0.0427955,0.058691,0.0784791,0.102154,0.130609,0.162303,  0.2001,0.241264,0.291239, 0.34394,0.407206,0.476039],[0.0299085,0.0427915,0.0586815,0.0784864,0.101918,0.130551,0.162318,0.200092,0.241282,0.291222,0.344018,0.407216,0.476047],[0.0299115,0.0427963,0.0586865,0.0784855,0.102157,0.130595,0.162316,0.200104,0.241289,0.291197,0.344148,0.407937, 0.47603],[0.029915,0.042795,0.0586821,0.078478,0.102227,0.130574, 0.16232,0.200098,0.241292,0.291229,0.344093,0.407185,0.476066],[0.0299118,0.0428075,0.0586845,0.0785065,0.102169,0.130559,0.162325,0.200111,0.241313,0.291206,0.344066,0.407192,0.476061],[0.029914,0.042756,0.0586935,0.078475,0.102132,0.130577,0.162339,0.200096,0.241292,0.290953,0.344134,0.407224,0.476242],[0.029911,0.042793,0.0586925,0.0784975,0.102142,0.130589,0.162299,0.200092,0.241302,0.291207,0.344077,0.407217,0.476052],[ 0.02991,0.0427988,0.058692,0.0784885,0.101915,0.130567,0.162318,0.200112,0.241318,0.291223, 0.34404,0.407207,0.476024],[0.0299075,0.0427508,0.0586885,0.0784831, 0.10201,0.130561,0.162299,0.200101, 0.24132,0.291203,0.344197,0.407198,0.476024],[0.0299145,0.0427768,0.0586934,0.0784855,0.102144,0.130561,0.162292,0.200108,0.241299,0.291155,0.344105,0.407232,0.476023],[0.029909,0.0427977,0.0586849,0.078493,0.102218,0.130597,0.162316,0.200136,0.241289,0.291226,0.344073,0.407235,0.476006],[0.0299203,0.0427995,0.0586845,0.0785301,0.102158,0.130553,0.162291,0.200103,0.241302,0.291233,0.344172,0.407235,0.476058]]

Mflops_row = \
[[0.0364143,0.0521575,0.0713634,0.0955125,0.123734, 0.15885, 0.19731,0.242694,0.291539,0.352865,0.417255, 0.52724, 0.71171],[0.0364175,0.0521746,0.0713515,0.095521,0.123579,0.158183,0.197387,0.242673,0.291542,0.352123,0.417195,0.527132,0.711155],[0.0364107,0.0521541, 0.07135,0.0955189,0.123551,0.158157,0.197322,0.242678,0.291517,0.352114,0.417202,0.527476,0.711644],[0.036405, 0.05217,0.071349,0.0955011,0.123563,0.158166,0.197316,0.242659,0.291545,0.352106,0.417206,0.527386, 0.71093],[0.0364113,0.0521605,0.0713531,0.0954895,0.123566,0.158156,0.197301,0.242693,0.291561,0.352104,0.417226, 0.52717,0.711321],[0.0364112,0.0521539,0.071354,0.095546,0.123583, 0.15817,0.197338,0.242648,0.291586,0.352146,0.417213, 0.52727, 0.71097],[0.0364112,0.0521749,0.0713509,0.0955031,0.123592,0.158175,0.197311,0.242671, 0.29154,0.352115,0.417176,0.526795,0.710827],[0.036423,0.0521575,0.071354,0.095504,0.123559,0.158152,0.197319,0.242722,0.291538, 0.35215,0.417208,0.527012,0.710964],[0.0364185,0.0521821,0.0713565,0.0955056,0.123562, 0.15815,0.197328,0.242657,0.291541,0.352124,0.417193,0.527332,0.711018],[0.036409,0.0521735,0.071384,0.095503,0.123573,0.158158,0.197293,0.242663,0.291584, 0.35226,0.417225,0.527276,0.710977],[0.0364085,0.052163,0.0713495,0.0954995,0.123566,0.158161,0.197328,0.242702,0.291553,  0.3521,0.417222,0.527076,0.710964],[0.0364133, 0.05217,0.0713505,0.0955375,0.123578,0.158165,0.197307,0.242678,0.291527,0.352123,0.417179,0.527107,0.711015],[0.0364095,0.0521646,0.0713515,0.0955105,0.123578,0.158157,0.197303, 0.24268,0.291518,0.352087,0.417187,0.527337,0.711097],[0.0364093,0.0521635,0.071345,0.0954995,0.123589,0.158204,0.197328,0.242677,0.291552,0.352104,0.417206,0.527515,0.710906],[0.0364052,0.0521644,0.071353,0.0955175,0.123575,0.158151,0.197363, 0.24267, 0.29154,0.352086,0.417229,0.527408,0.710932],[0.036407,0.0521564,0.0713495, 0.09552,0.123591, 0.15817, 0.19733,0.242678,0.291541,0.352152,0.417181,0.526983,0.710996],[0.0364068,0.0521555,0.071352,0.0954895,0.123571,0.158168,0.197358,0.242684,0.291537,0.352108,0.417195,0.527058,0.710663],[0.0364068,0.0521641,0.071342,0.095506,0.123569, 0.15821,0.197306,0.242694,0.291583,0.352147,0.417199,0.527466,0.711057],[0.0364167,0.0521621,0.0713525,0.0955006,0.123579,0.158168,0.197329,0.242681, 0.29159,0.352124,  0.4172,0.527184,0.711225],[ 0.03641,0.0521625,0.0713514,0.0955055,0.123559,0.158162,0.197331,0.242681,0.291547,0.352132,0.417279,0.526936,0.711186]]

Mflops_blas = \
[[0.426222,0.606896,0.831171, 1.10661, 1.43559, 1.82581, 2.27965,   2.804, 3.40077, 4.07863, 4.84088, 5.72172, 6.73448],[0.426236,0.606508,0.831126, 1.10672, 1.43598, 1.82579, 2.27961,  2.8039,  3.4006, 4.07862, 4.84208, 5.72135, 6.73209],[0.426224,0.606556,0.831179, 1.10656, 1.43553, 1.82586, 2.27971, 2.80404, 3.40071, 4.07852, 4.84073, 5.72152, 6.73235],[ 0.42623,0.606493,0.831165, 1.10649, 1.43541,  1.8258, 2.27965, 2.80409, 3.40054, 4.07868, 4.84102,  5.7215,  6.7322],[0.426249,0.606512,0.831189, 1.10656, 1.43573, 1.82595, 2.27972, 2.80393, 3.40063, 4.07843,  4.8409, 5.72202, 6.73361],[0.426207,0.606474,0.831096,  1.1065, 1.43539,  1.8258, 2.27987, 2.80388,  3.4007, 4.07906, 4.84078, 5.72303, 6.73231],[0.426253,0.606578,0.831209, 1.10685, 1.43553, 1.82579, 2.27962, 2.80397, 3.40058,  4.0786, 4.84162, 5.72148, 6.73225],[0.426267,0.606536,0.831116,  1.1066, 1.43544, 1.82586, 2.27969, 2.80418, 3.40065, 4.07847, 4.84081, 5.72171, 6.73224],[ 0.42621,0.606516,0.831102, 1.10655, 1.43541, 1.82576, 2.27958, 2.80403, 3.40066, 4.07872, 4.84089,  5.7216, 6.73207],[0.426332,0.606553,0.831195,  1.1071, 1.43557, 1.82578, 2.27968, 2.80456, 3.40062, 4.07844, 4.84091, 5.72186, 6.73288],[0.426259,0.606493, 0.83112,  1.1065, 1.43557, 1.82578, 2.27962, 2.80403, 3.40081, 4.07847, 4.84092, 5.72263, 6.73228],[0.426226,0.606568,0.831157,  1.1068, 1.43546, 1.82588, 2.27969, 2.80397, 3.40054, 4.07858, 4.84107, 5.72136, 6.73207],[0.426191,0.606505,0.831172, 1.10652, 1.43552,  1.8259,  2.2797, 2.80405, 3.40116, 4.07839, 4.84079, 5.72193, 6.73255],[0.426222,0.606509, 0.83115, 1.10652, 1.43551, 1.82585, 2.28021, 2.80399,  3.4007, 4.07844, 4.84082,  5.7214, 6.73237],[0.426268, 0.60656, 0.83161, 1.10679, 1.43559, 1.82585,  2.2796, 2.80384, 3.40055, 4.07838,   4.841, 5.72205, 6.73235],[0.426229,0.606521,0.831109, 1.10648, 1.43545, 1.82586, 2.27955, 2.80413, 3.40081,  4.0784, 4.84098, 5.72186, 6.73226],[0.426222, 0.60657,0.831133, 1.10649, 1.43539, 1.82583,  2.2796, 2.80391, 3.40048, 4.07856, 4.84115, 5.72131,  6.7319],[0.426216,0.606542,0.831171, 1.10656,  1.4355, 1.82577, 2.27958, 2.80405, 3.40107, 4.07837, 4.84077, 5.72152, 6.73208],[0.426206,0.606496,0.831135, 1.10658, 1.43566, 1.82581, 2.27971, 2.80391, 3.40072, 4.07864, 4.84082, 5.72154,  6.7322],[0.426261,0.606987,0.831324, 1.10661, 1.43552,  1.8259, 2.27957, 2.80391, 3.40058, 4.07843, 4.84109,   5.722, 6.73186]]
