test_sizes = [550,900,650,800,500,1000,950,700,400,600,750,850,450]

Mflops_julia = \
[[0.00490625,0.035,0.0078125,0.01425,0.00378125,0.03975,0.023,0.0095625,0.00184375,0.00625,0.0116875,0.03,0.002671875],[0.00484375,0.019625,0.0079375,0.03025,0.00371875,0.02725,0.023125,0.0095625,0.00184375,0.00621875,0.0116875,0.017125,0.002671875],[0.00484375,0.02025,0.00775,0.02625,0.00371875,0.02675,0.02325,0.0096875,0.001875,0.00625,0.011625,0.02775,0.002671875],[0.004875,0.03275,0.0079375,0.02625,0.00371875,0.02725,0.023,0.0095625,0.001859375,0.00621875,0.0116875,0.017125,0.0026875],[0.00484375,0.019875,0.0078125,0.02525,0.00371875,0.02675,0.023375,0.0095625,0.001875,0.00621875,0.011625,0.02775,0.0026875],[0.00484375,0.0325,0.0078125,0.014125,0.00375,0.0265,0.023375,0.009625,0.001859375,0.00625,0.0116875,0.02775,0.0026875],[0.004875,0.0325,0.0078125,0.026,0.00375,0.027,0.023125,0.009625,0.001859375,0.00621875,0.0116875,0.016875,0.0026875],[0.00484375,0.01975,0.0078125,0.014625,0.00371875,0.02675,0.023,0.009625,0.001875,0.00621875,0.0116875,0.02775,0.002671875],[0.00484375,0.0325,0.00775,0.014125,0.00371875,0.028,0.023375,0.009625,0.001875,0.00621875,0.011625,0.028,0.0026875],[0.004875,0.03275,0.0078125,0.0265,0.00371875,0.02725,0.023125,0.009625,0.001875,0.00625,0.0116875,0.017125,0.002671875],[0.004875,0.01975,0.00775,0.02525,0.00371875,0.027,0.0235,0.0095625,0.001875,0.00621875,0.011625,0.02775,0.002671875],[0.00484375,0.03275,0.0078125,0.02525,0.00375,0.027,0.023375,0.0095625,0.001875,0.0063125,0.0116875,0.017,0.002703125],[0.00484375,0.01975,0.00775,0.014375,0.00371875,0.027,0.023125,0.009625,0.001890625,0.00621875,0.011625,0.02775,0.0026875],[0.00646875,0.033,0.007875,0.02525,0.00375,0.027,0.023125,0.009625,0.001875,0.0063125,0.0116875,0.017125,0.0026875],[0.00503125,0.019625,0.00775,0.014625,0.00375,0.027,0.023,0.0095625,0.001875,0.0063125,0.0116875,0.01675,0.002640625],[0.0048125,0.0325,0.0078125,0.02525,0.00375,0.02675,0.023125,0.009625,0.001859375,0.00625,0.011625,0.016875,0.0026875],[0.004875,0.019625,0.00775,0.014625,0.00371875,0.027,0.023,0.009625,0.001859375,0.0063125,0.0116875,0.01675,0.00265625],[0.00484375,0.0325,0.0078125,0.014125,0.00378125,0.0265,0.023375,0.009625,0.001890625,0.00625,0.0116875,0.02775,0.002703125],[0.004875,0.03375,0.0078125,0.02525,0.0038125,0.027,0.02325,0.009625,0.001859375,0.00625,0.012,0.017125,0.002703125],[0.00484375,0.01975,0.0078125,0.014625,0.00378125,0.02675,0.023,0.009625,0.001890625,0.0063125,0.01175,0.016875,0.002671875]]

Mflops_genvect = \
[[0.041599,0.188657,0.0701585,0.128018,0.031039,0.253708,0.221417,0.0870625,0.0159699,0.0540795,0.106442,0.154988,0.0232504],[0.0416343,0.188614,0.0702125, 0.12795,0.0310467,0.253706,0.221559,0.087099,0.0159756,0.054089,0.106255, 0.15482,0.0232434],[0.0416305,0.188746,0.070164,0.128026,0.031069,0.253621,0.221412,0.087085, 0.01599,0.054086,0.106218,0.155069,0.023259],[0.0416228,0.188519,0.0701615,0.127998,0.031067,0.253727,0.221514,0.0870605,0.0159679,0.0540944,0.106213,0.155034,0.0232445],[0.0415805,0.188628,0.0701715,0.127944,0.0310727,0.253732, 0.22157,0.087086,0.0159735,0.054175,0.106234,0.154842,0.0232543],[0.041634,0.188503,0.0701935,0.127907,0.0310493,0.253715,0.222184,0.087057,0.0159656,0.0540495,0.106268,0.154942,0.023241],[0.041589,0.188434,0.0701411,0.127956,0.0310575,0.253686, 0.22135,0.0870825,0.0159658,0.054099,0.106355,0.154882,0.0232439],[0.0416192,0.188647,0.0701655,0.128121,0.0310553,0.253786,0.221512,0.0870895,0.015972,0.0540955,0.106222, 0.15496,0.023248],[ 0.04162,0.188562,0.0702164,0.128095,0.0310742,0.253725,0.221496,0.0870715,0.0159676,0.0541041,0.106307,0.155728,0.0232509],[0.0416207,  0.1886,0.0701574,0.128007,0.031061,0.253679,0.221458,0.087074,0.0160875,0.0540855,0.106248,0.154922,0.0232514],[0.0416075,0.188852,0.0701605,0.127951,0.031042,0.253739,0.221656,0.0870675,0.0159654,0.0541055,0.106224,0.154835,0.0232483],[0.041593,0.188698,0.070159,0.128173,0.0310553,0.253745,0.221509,0.0870501,0.0159672,0.0540975,0.106257,0.154875,0.023248],[0.0416098,0.188549,0.0701995,   0.128,0.0310385,0.253577,0.221406,0.087072,0.0159692,0.0540484,0.106221, 0.15483,0.0232528],[0.0416138,0.188674,0.070135,0.127904,0.031055,0.253633,0.221578,0.087027,0.0159646,0.054101,0.106235,0.155044,0.0232486],[0.0415927,0.188623,0.0701455,0.128005,0.0310472, 0.25364,0.221798,0.087054,0.0159701,0.0540835,0.106321,0.154913,0.0232382],[0.0416088,0.188813,0.0701694, 0.12801,0.0310737,0.253784,0.221576,0.0870864,0.0159744,0.0540899,0.106288,0.154927,0.0232479],[0.0416125,0.189334,0.0701605,   0.128,0.0310695,0.253643,0.221427,0.0870484,0.0159709,0.0541235,0.106287,0.154952,0.0232602],[0.0416032,0.188745,0.070172, 0.12796,0.0310413,0.253704,0.221465,0.087091,0.0159734,0.0541015,0.106357,0.154829,0.023242],[0.0416023,0.188707,0.0702025,0.128029,0.0310477,0.253901, 0.22149,0.0870955,0.0159732,0.0540515,0.106299,0.154969,0.0232429],[0.0418652,0.188497,0.0702076,0.128069,0.0310475,0.253639, 0.22154,0.0870634,0.015966,0.0540766,0.106197,0.155105,0.0232524]]

Mflops_autovect = \
[[0.071877,0.313861,0.118509,0.218136,0.0542104,0.431417, 0.36922,0.147588,0.0272868,0.092249,0.181603,0.263321,0.0393825],[0.071828,0.313632,0.118534,0.218226,0.0540695,0.431331,  0.3692,0.147478,0.0272867,0.0922354,0.181463,0.263307,0.0393975],[0.0717875,0.313726, 0.11849,0.218257, 0.05407,0.431481, 0.36919,0.147494,0.0273065,0.092453,0.181586,0.263376,0.0394002],[0.0718325,0.313772,0.118495,0.218189,0.0541195,0.431256,0.369045, 0.14752,0.027294,0.092483,0.181879, 0.26346,0.0394005],[0.071777,0.313804,0.118491,0.218304,0.0540925,0.431506,0.369503,0.147514,0.0273003,0.0922564,0.181565,0.263459,0.0394045],[0.0718435,0.313871,0.118574,0.218282,0.054058,0.431736,0.369252,0.147512,0.027295,0.0922816,0.181573, 0.26349,0.039418],[0.0718315,0.313637,0.118504,0.218294,0.054077,0.431497,0.369471,0.147538,0.0273158,0.092239,0.181478,0.263372,0.0393682],[0.0717885,0.313736,0.118466,0.218277,0.0540335,0.431438,0.369204,0.147429,0.027307,0.0922985,0.181459,0.263374,0.0393702],[0.0718089,0.313705,0.118537,0.218201,0.0540659,0.431488,0.369299, 0.14758,0.0273075,0.092265,0.181562, 0.26343,0.0393625],[0.0718125,0.313973,0.118542,0.218239,0.054045,0.431358,0.369291,0.147485,0.0273041, 0.09224,0.181601,0.263266,0.0393747],[0.071833, 0.31368,0.118522,0.218208,0.0540495, 0.43139,0.369583,0.147429,0.0273125,0.0922351,0.181499,0.263341,0.0393785],[0.072039,0.313635,0.118488,0.218323,0.0540434,0.431422,0.369021,0.147443,0.027308,0.0922251,0.181564,0.263282,0.0393867],[0.0719205, 0.31363,0.118448,0.218184,0.054038,0.431479,0.369443,0.147521,0.0273103,0.092234,0.181504,0.263371,0.0393948],[0.071937,0.313693,0.118559,0.218326,0.054063,0.431442,0.369153,0.147488,0.0273335,0.092204,0.181557,0.263467,0.0393673],[0.0718274,0.313722,0.118562,  0.2183,0.0540435,0.431358,0.369223,0.147453, 0.02732,0.0921675,0.181467, 0.26334,0.0393745],[0.071865,0.313696,0.118697,0.218216,0.054045,0.431426,0.369139,0.147493,0.0273063,0.0922329,0.181567,0.263386,0.0393755],[0.0718855,0.313647,0.118514,0.218151,0.054045,0.431338,0.369249, 0.14753,0.0273065,0.092163,0.181545,0.263291,0.0393817],[0.0718305,0.313762, 0.11849,0.218252,0.0541259,0.431565,0.369107,0.147449,0.0273228,0.092234,0.181495,0.263368,0.0393588],[0.0718515,0.313616,0.118501,0.218297,0.0540404,0.431271,0.369485,0.147525,0.0273045,0.092201, 0.18158,0.263314,0.0393655],[0.0718526,0.313773, 0.11854,0.218217,0.0541029, 0.43144,0.369278,0.147508,0.0273065,0.092162,0.181556,0.263474,0.0394022]]

Mflops_basic = \
[[0.266386, 1.16564,0.437259,0.767154,0.179556, 1.80334, 1.39166,0.549007,0.0975929,0.349841,0.673879,0.976515,0.109719],[0.266382, 1.16524,0.437069,0.767125,0.179513, 1.80973, 1.39306,0.548983,0.0976125,0.349763,0.673779,0.976568,0.110558],[ 0.26634, 1.16572,0.437219,0.767131,0.180062, 1.81154, 1.39641,0.549678,0.0976416,0.349892,0.674142,0.976698,0.109688],[0.266334, 1.16568, 0.43708,0.767287,0.179539, 1.80688, 1.39344,0.549031,0.0976241,0.349821,0.673923,0.976515,0.109666],[0.266262, 1.16519,0.438398,0.767089,0.179553, 1.80686, 1.39312,0.549109,0.0976651,0.349796,0.673999,0.976843,0.109747],[0.266389, 1.16546,0.437191,0.767297,0.181169, 1.80716, 1.39247,0.549136,0.0975901,0.349817,0.674506,0.976572,0.109857],[0.266274, 1.16519,0.437351,0.767153,0.179597, 1.81187, 1.39429,0.549062,0.097659,0.349673,0.673997,0.977478,0.109724],[0.266284, 1.16508,0.437155,0.767211,0.179651, 1.81002,  1.3948,0.549138,0.0976695,0.349781,0.673725,0.976811,0.109663],[0.266295, 1.16544,0.437112,0.767254,0.179516, 1.80405, 1.39691,0.548956,0.097625,0.349715,0.673987, 0.97651,0.109723],[ 0.26625, 1.16534,0.437377,0.767209,0.179555, 1.80602, 1.39483,0.549137,0.097683,0.349896,0.674037,0.976921, 0.10975],[0.266556, 1.16542, 0.43733,0.767147,0.180929, 1.80401, 1.39255,0.548915,0.097614,0.349724,0.676096,0.976567,0.109712],[0.266234, 1.16533,0.437193,0.767132,0.179602, 1.80737, 1.39332,0.548971,0.0976175,0.349789,0.673842,0.978941,0.109956],[0.266994, 1.16535,0.437202,0.767022,0.179625, 1.80716, 1.39281,0.549011,0.097635, 0.35065,0.674104,0.976757,0.109973],[0.266447, 1.16562, 0.43729,0.767228,0.179859,  1.8058,  1.3944,0.548978,0.0976095,0.349709,0.673839, 0.97654,0.109717],[0.266311, 1.16543,0.438451,0.766916,0.179539, 1.81051, 1.39215,0.549027,0.097596, 0.34985,0.673846,0.976891,0.109702],[ 0.26644,  1.1656,0.437225,0.767745,0.179706, 1.80811, 1.39208,0.549053,0.0978774,0.349685,0.674415, 0.97644,0.109728],[0.266243,  1.1655,0.437253,0.767144,0.179525, 1.80888,   1.394,0.549046,0.097603,0.349783,0.674014,0.977357, 0.10968],[0.267328, 1.16507,  0.4371,0.766994,0.179498, 1.80696, 1.39488,0.549049,0.097607,0.350697,0.673953,0.976505,0.109945],[0.266376, 1.16738,0.437068,0.766935,0.179492, 1.80384, 1.39629, 0.54892,0.097651,0.349829,0.673712,0.976466,0.109998],[0.266382, 1.16524,0.437159,0.767147,0.179517, 1.80863,  1.3936,0.550561,0.0976175,0.350117,0.674072,0.976855,0.109756]]

Mflops_blocked = \
[[0.079139,0.350765,0.130714, 0.24018,0.0596069,0.476507,0.412659,0.164363,0.030231,0.102348,0.199998, 0.29196,0.0433773],[0.0791715,0.351166,0.130655,0.240088,0.059629,0.476501,0.412087,0.164479,0.0302578,0.102384,0.200057,0.292012,0.0433415],[0.0791456,0.350679, 0.13067,0.240152,0.0595859,0.476601,0.412194,0.164366,0.030229,0.102369,0.200169,0.291944,0.0433227],[0.0791415,0.350721,0.130666,0.240177,0.059582,0.476851,0.412642,0.164403,0.0302272,0.102331,0.200067,0.291855,0.043333],[0.0791534,0.350924,0.130776,0.240219,0.059584,0.476613,0.411965,0.164436,0.030246,0.102327,0.200098,0.291928,0.0433242],[0.079188,0.350854,0.130773,0.240167,0.0596479,0.476734,0.412138,0.164476,0.0302605,0.102363,0.200068,0.292014,0.0433415],[0.0792345,0.350648,0.130747,0.240205,0.0595785,0.476744,0.412224,0.164463,0.0302425,0.102347,0.200071,0.291894,0.0433375],[0.0791974,0.350677,0.130739,0.240414,0.0595886,0.476612,0.412225,0.164442,0.030228,0.102323,0.200083,0.292007,0.0433205],[0.0791764,0.350719,0.130786,0.240152,0.059602, 0.47661,0.412147,0.164388,0.0302498,0.102323,0.200104,0.291915,0.043328],[0.079154,0.350704,0.130718,0.240208,0.059589,0.476667,0.412181,0.164437,0.030245,0.102383,0.200075, 0.29192, 0.04334],[0.079146,0.350787,0.130699,0.240258,0.0595856,0.476859,0.412078,0.164354,0.0303432,0.102459,0.199962,0.291855,0.0433277],[0.0791731,0.350827,0.130674,0.240154,0.0595729,0.476679,0.412046,0.164386,0.0302318,0.102371,0.200018,0.291903,0.0433357],[0.0791745,0.350755,0.130791,0.240147,0.059618,0.476773, 0.41205,0.164502,0.0302613, 0.10241,0.200242,0.292016,0.0433432],[0.0791481,0.350797,0.130664,0.240233,0.0595781,0.476595,0.413133,0.164428,0.0302353,0.102409,0.200065,0.291976,0.0433255],[0.079165,0.350746,0.130756,0.240218,0.0595959,0.476542,0.412281,0.164471,0.0302472,0.102411,0.199989,0.291962,0.0433468],[0.079163,0.350979,0.130741,0.240238, 0.05959, 0.47669,0.412058,0.164513,0.0302308,0.102333,0.200011,0.291875,0.0433233],[0.079199,0.350774,0.130691,  0.2401,0.059563, 0.47665,0.412089,0.164433,0.030284,0.102371, 0.20014, 0.29196,0.0433232],[0.079191,0.350686,0.130741,0.240201,0.0596075,0.476682,0.412022,0.164359,0.030261,0.102352,0.200049,0.291962, 0.04332],[0.0792375,0.350826,0.130666,0.240219,0.059587,0.476848,0.412124,0.164373,0.0302615,0.102329, 0.20006,0.291904,0.0433135],[0.079199,0.350827,0.130666,0.240344,0.059605,0.476603,0.412196,0.164362,0.0302415,0.102409,0.200133,0.292023,0.0433232]]

Mflops_col = \
[[ 0.51021, 2.22897,0.839675, 1.74524,0.385182, 3.32123,  2.6498, 1.05156,0.198305,0.674396, 1.30711,  1.8755,0.280652],[0.510615, 2.22642,0.839804, 1.74554,0.385091,  3.3297, 2.64883, 1.05249,0.198287,0.674277, 1.30664, 1.87571,0.280721],[0.510748, 2.22423,0.839699, 1.74667,0.385144, 3.32202, 2.65099, 1.05159,0.198388,0.674373, 1.30675, 1.87584,0.280746],[0.510184, 2.22456,0.839777, 1.74552,0.385065, 3.32239, 2.65243, 1.05162, 0.19829,0.674368, 1.30675,  1.8764,0.280625],[0.510148, 2.22415,0.839749, 1.74727,0.385091, 3.32051, 2.65373, 1.05167,0.198243,0.674359,  1.3068, 1.87691,0.280685],[0.510411, 2.22442,0.839724, 1.74516,0.385092, 3.32241, 2.65138, 1.05154,0.198314,0.674455, 1.30661, 1.87567, 0.28065],[0.510226, 2.22537,0.839587, 1.74616,0.385101,  3.3272, 2.65553, 1.05181,0.198318,0.674325, 1.30674, 1.87682, 0.28074],[0.510187,  2.2244,0.839571, 1.74579,0.385272, 3.32176, 2.65015, 1.05173,0.198282,0.674529, 1.30681,  1.8755,0.281652],[0.510219, 2.22645,0.839824, 1.74561,0.385157, 3.32329, 2.64857, 1.05154,0.198236,0.674227, 1.30664, 1.87553,0.280753],[0.510172, 2.22461,0.839817, 1.74841,0.385204, 3.32264, 2.65037, 1.05154,0.198238,0.674446, 1.30665, 1.87603,0.280694],[0.510295, 2.22442,0.839983,  1.7456,0.385128, 3.32709, 2.65595, 1.05182,0.198254, 0.67431, 1.30678, 1.87572,0.280794],[0.510061,  2.2243,0.839623, 1.74551,0.385148, 3.32265, 2.65463, 1.05187,0.198274,0.674447, 1.30712, 1.87598,0.280689],[ 0.51026, 2.22438, 0.83961, 1.74517,0.385104, 3.32251, 2.64965, 1.05154,0.198191,0.675431, 1.30649, 1.87576,0.280766],[0.510329, 2.22513,0.839475, 1.74524,0.385124, 3.32467, 2.65225,  1.0516,0.198263,0.674555, 1.30663, 1.87705,0.280727],[0.510105, 2.22597,0.839602, 1.74545,0.385183, 3.32557, 2.65075, 1.05158,0.198254,0.674403, 1.30712, 1.87536,0.280703],[0.510159, 2.22631,0.839876, 1.74532,0.385114, 3.32771, 2.65172, 1.05181,0.198244,0.674366, 1.30875,  1.8756,0.280665],[0.510228, 2.22531,0.839631, 1.74649,0.385126, 3.32186, 2.65024, 1.05173, 0.19825, 0.67451, 1.30673, 1.87625,0.280747],[0.510257, 2.22486,0.839994, 1.74549,0.385264, 3.32227, 2.65068, 1.05158,0.198385,0.674466, 1.30677, 1.87561, 0.28064],[0.510203, 2.22452,0.839826, 1.74607,0.385117, 3.32826, 2.65307, 1.05192, 0.19833,0.674592, 1.30696,  1.8757,0.280732],[0.510175, 2.22451,0.840572, 1.74515,0.385047,  3.3227, 2.65445, 1.05181,0.198289,0.675989, 1.30677, 1.87559, 0.28074]]

Mflops_copy = \
[[0.0783194,0.344309,0.129533, 0.24044,0.058719,0.473375,0.404177,0.161208,0.0299867,0.101908,0.198437,0.288828,0.0427907],[ 0.07836,0.344505,0.129522,0.240454,0.0586821,0.473153,0.404188,0.161232,0.0299935,0.102188,0.198441, 0.28847,0.0427853],[0.0785955,0.344385,0.129496,0.240407,0.0587155,0.473375, 0.40422,0.161334,0.0299935,0.101876,0.198498,0.288559,0.0427938],[0.078339,0.344382,0.129613,0.240391,0.0587455,0.473227,0.404487,0.161303,    0.03,0.102118,0.198404,0.288915,0.0427925],[0.078325,0.344256,0.129531,0.240494,0.0587649,0.473384, 0.40433,0.161232,0.0300083,0.101905,0.198446,0.288829,0.0427812],[0.0783451,0.344128,0.129755,0.240477,0.0587164,0.473136,0.404259,0.161254,    0.03,0.102125,0.198453,0.288935,0.0427925],[0.0783875,0.344147,0.129537,0.240463,0.0587461,0.473224,0.404232,0.161273,0.0299815,0.102123,0.198465,0.288736,0.042786],[0.0783376,0.344302,0.129546,0.240427,0.0587161,0.473104, 0.40448,0.161209,0.029994,0.101988,0.198479,0.288822,0.0427835],[0.0783911,0.344542,0.129525,0.240498,0.0587075,0.473189,0.404189,0.161247,0.0299845,0.102121,0.198482,0.288516,0.0428255],[0.0783144, 0.34432,0.129531,0.240454,0.058728,0.473223,0.404238,0.161212,0.0299995,0.102848,0.198455,0.288476,0.0427802],[0.0783365, 0.34448, 0.12977, 0.24041,0.0586795, 0.47318,0.404211,0.161212,0.0299857,0.102051,0.198481,0.289276,0.0427877],[0.0784445,0.344365,0.129571,0.240416,0.058697,0.473286,0.404215,0.161309,0.0299847,0.102163,0.198473,0.288868,0.0428162],[0.0784215,0.344395,0.129555,0.240505,0.058724,0.473313,0.404209,0.161349,0.0299913,0.102159,0.198475,0.288507,0.0427942],[0.078364,0.345289,0.129581,0.240459, 0.05879,0.473276,0.404272,0.161325,0.0299928,0.101862,0.198429,0.289041,0.0427735],[  0.0784,0.344216,0.129562,0.240461,0.058697,0.474197,0.404241,0.161282,0.0301207,0.102161,0.198417, 0.28891,0.0427982],[0.0783135,0.344243,0.129515,0.240529,0.058708,0.473149,0.404221, 0.16125,0.0299838,0.102083,0.198465,0.288547,0.0427975],[0.0783559,0.344362, 0.12954,0.240467,0.058748,0.473173,0.404266,0.161248,0.029985,0.102089,0.198451,0.288965,0.042804],[0.0785615,0.344317,0.129576,0.240427,0.0586671,0.473327,0.404316,0.161231,0.0299865,0.102162,0.198458,0.289115,0.0427758],[0.078371,0.344321,0.129522,0.240483,0.0586745,0.473299,0.404195,0.161238,0.0300005,0.102158,0.198502,0.288622,0.0427805],[0.0783961,0.344381,0.129562,0.240447,0.0587455,0.473359,0.404238,0.161504, 0.02999,0.102161,0.198504,0.288861,0.0428085]]

Mflops_naive = \
[[0.281222, 1.23095,0.463574,0.818153,0.187364, 1.95761, 1.48032,0.579154,0.103674,0.365025,0.715433,  1.0361,0.119509],[0.281111, 1.23031,0.463443,0.817442,0.187407, 1.95415, 1.48843,0.579031,0.103678,0.365345, 0.71577, 1.03649, 0.11957],[0.281016, 1.23714,0.463511,0.817583,0.187371, 1.95592, 1.48195,0.578963, 0.10366,0.365029,0.715546, 1.03599,0.119539],[0.281024, 1.23063,0.463742,0.817583,0.187409, 1.95439, 1.48184,0.580483,0.103676,0.365499,0.715561,  1.0361, 0.11958],[0.281008, 1.23093, 0.46346,0.817507,0.188098, 1.95881, 1.48148,0.579033,0.103696,0.365674, 0.71556, 1.03643,0.119523],[0.281027, 1.23054,0.463414,0.817633,0.187373, 1.95651, 1.48239,0.578923,0.103722,0.365035,0.717891, 1.03615,0.119548],[0.281043, 1.23029,0.463588,  0.8176,0.187351, 1.96229, 1.48515,0.579045,0.103725,0.365084,0.715598, 1.03639,0.119574],[0.281146, 1.23048,0.463363,0.817551,0.187278, 1.95564, 1.47853,0.578999,0.103714,0.365041,0.715371, 1.03904,0.119543],[ 0.28102, 1.23064,0.463456,0.817406,0.187374, 1.95875, 1.48111,0.578949,0.103679, 0.36529,0.715597, 1.03619,0.119618],[0.281019, 1.23104,0.463557,0.820368,0.187328, 1.96144, 1.48109,0.578963,0.103671, 0.36518,0.715633, 1.03621,0.119604],[0.281036, 1.23021,0.463443, 0.81765,0.187297, 1.95636, 1.48282,0.579002,0.103667, 0.36502,0.715533, 1.03835,0.119506],[0.280964,  1.2334,0.463362,0.817462,0.187293, 1.96176, 1.49009,0.579059,0.103677, 0.36513,0.715456, 1.03634,0.119551],[0.281085, 1.23037, 0.46522,0.817267, 0.18744, 1.95063, 1.48361,0.579501,0.103676,0.365046,0.715611, 1.03624, 0.11955],[0.281017, 1.23026,0.463343,0.819572,0.187339, 1.95415, 1.48121,0.579012,0.103661,0.365098,0.716464, 1.03614,0.119513],[0.281022, 1.23069,0.463258,0.817514,0.188331, 1.95525, 1.47955,0.578951,0.103697,0.365047,0.716423, 1.03632,0.119518],[0.281123,  1.2307,0.463336,0.817364,0.187311,  1.9537,  1.4781,0.579132,0.103724,0.365072,0.715571, 1.03622,0.119571],[0.281068, 1.23067,0.463445,0.817615,0.187341, 1.95975, 1.48124,0.578884,0.103666,0.364972,0.715608, 1.03757,0.119533],[0.281067, 1.23034, 0.46328,0.817455,0.187351, 1.95641, 1.48002,0.579174,0.103677,0.365091,0.715413, 1.03633,0.119509],[0.281042,  1.2305,0.463553,0.817361,0.187347, 1.95257, 1.48075,0.578928,0.103698,0.364974,0.715453,  1.0365,0.119558],[0.281025, 1.23037,0.463428,0.817454, 0.18736, 1.95372, 1.48315,0.579015,0.103675,0.365086,0.715405, 1.03587,0.119605]]

Mflops_rb = \
[[0.0784745,0.343245,0.130444, 0.24092,0.0585746,0.476079,0.406528,0.162227,0.0299215,0.101913,0.199813,0.290808,0.0427027],[0.0783834,0.343626,0.130405, 0.24096,0.0586511, 0.47579, 0.40645,0.162207,0.0299333, 0.10202,0.199836,0.290924,0.0426987],[0.078974,0.343541,0.130457,0.240925,0.0586369,0.475763,0.406366, 0.16227,0.029941,0.102083,0.199861,0.290679,0.0427015],[0.078414,0.343581,0.130541,0.240921,0.0586265,0.475796,0.407157,0.162309,0.0299377,0.101998,0.199966,0.290982,0.0426985],[0.0784035,0.343194, 0.13046, 0.24101,0.058623,0.475977,0.406529,0.162168,0.0299208,0.102146, 0.19988,0.290798,0.0426828],[0.0783455,  0.3435,0.130647,0.240907,0.0586081,0.475873,0.406291,0.162164,0.029926,0.102002,0.199836,0.290678,0.0426873],[0.078385,0.343739,0.130465, 0.24087,0.058655,0.475882,0.406273,0.162269,0.029964,0.102098,0.199893,0.291009,0.042685],[0.078378,0.343327,0.130451,0.240987,0.0586009,0.475838,0.406319,0.162345,0.029944,0.101996,0.199872,0.291019,0.0426937],[0.0784245,0.343183,0.130428,0.240916,0.058607,0.475873,0.406207,0.162228,0.0299245, 0.10191,0.199868,0.290621,0.0427025],[0.0784149,0.343347, 0.13048,0.240886,0.0586005,0.475754,0.406364,0.162248,0.0299273,0.101965,0.199811,0.290886,0.0427117],[0.0784559,0.343861,0.130494,0.240947,0.0586239,0.475869, 0.40627,0.162194,0.0299383,0.102278,0.200059,0.290974,0.042693],[0.078421,0.343486,0.130405,0.241004,0.0585946, 0.47598,0.406348,0.162218,0.0299237,0.101996, 0.19983, 0.29093,0.0426885],[0.0783505,0.343725,0.130457,0.241053, 0.05863,0.476006,0.406265,0.162257,0.029935,0.102189,0.199889,0.290828,0.042684],[0.078396,0.344042,0.130456,0.241087,0.0585794,0.476007,0.406282,0.162262,0.0299258,0.101996,0.199827,0.290897,0.042671],[0.0784435,0.343861,0.130431,0.241053,0.0586045,0.475975,0.406172, 0.16225,0.0300538, 0.10205,0.199815,0.290848,0.0426933],[0.0783615, 0.34342,0.130449,0.240922,0.0586059,0.475845,0.406125,0.162358,0.0299222,0.101985,0.199863,0.290924,0.042694],[0.0785036,0.343625,0.130491,0.240998,0.058601,0.475968,0.406196,0.162213,0.0299287,0.102007,0.199873,0.290667,0.0426922],[0.0784994,0.343362,0.130468,0.241131,0.0586215,0.475957, 0.40642,  0.1622,0.0299402,0.101927,0.199929,0.291814,0.0426945],[0.0784526,0.343673,0.130408,0.240879,0.0586025,0.476239,0.406362,0.162251,0.0299327, 0.10199, 0.19987,0.290895,0.0426912],[0.078437,0.343498,0.130482,0.240907,0.0586045,0.476022,0.406381,0.162227,0.0299343,0.101995,0.199911,0.290626,0.0426955]]

Mflops_row = \
[[0.0955299,0.417635,0.158371,0.292282,0.0714509,0.715864,0.516211,0.197415,0.0364225,0.123789,0.242776,0.352671,0.0527205],[0.0955695,0.417595,0.158427,0.291652,0.071402,0.716742,0.516717,0.198098,0.0364185,0.123768,0.242881,0.352336,0.052228],[0.0956304,0.417721,0.158488,0.291794,0.0715696,0.714818,0.516649,0.197545,0.0364118, 0.12367,0.242857,0.352316,0.0524009],[0.0956395,0.417742,0.158393,0.291764,0.071373,0.719012,0.517168,0.197398,0.0364323,0.123701,0.242844,0.352265,0.0521765],[0.0955836,0.421072,0.158411,  0.2917,0.071372, 0.71462,0.517091,0.197513,0.036436,0.123795,0.242768,0.352249,0.0521905],[0.095607,0.418173,0.158476,0.291618,0.071376,0.716772,0.516191,0.197526,0.036419,0.123647,0.242845,0.352297,0.052166],[0.0956095,0.417693,0.158382, 0.29166,0.071363, 0.71557,0.515856,0.197463,0.0364273,0.123739,0.242794,0.352266,0.052153],[0.0955914,0.417673,0.158456,0.291702,0.0714295,0.715473,0.516127,0.197478,0.0364285,0.123707,0.242763,0.352211,0.052159],[0.0955944,0.417887,0.158348,0.291612,0.0714464,0.714294,0.516845,0.197473,0.0364275, 0.12381,0.243252,0.352314,0.0522056],[0.095546,0.417851,0.158353,0.291843,0.0714545,0.716198,0.517182,0.197587,0.0364398,0.123696,0.242841, 0.35242,0.0522529],[0.0956149,0.417909,0.158418, 0.29181,0.071365,0.716089, 0.51677,0.197503,0.036422,0.123671,0.242923,0.352322,0.052186],[0.0955515,0.417604,0.158401,0.291671,0.0713655,0.714022,0.515935,0.197569,0.0364208,0.123661,0.242773,0.352216,0.0521885],[0.095606,0.417472,0.158423,0.291727,0.0713915,0.715223, 0.51703,0.197519,0.0364205,0.123675, 0.24286,0.352249,0.0521785],[0.0956271,0.417584,0.158365,0.291692,0.0713761,0.714746,0.516394,0.197387,0.0364233,0.123758,0.242802,0.352219,0.0521555],[0.0960895,0.417751,0.158387,0.291668,0.0714484, 0.71361,0.516523,0.197415,0.0364243,0.123699,0.242874,0.352318,0.052177],[0.0956295,0.418628,0.158414,0.291725,0.071787,0.715226,0.516764,0.197406,0.0364215,0.123766, 0.24404,0.352357,0.0522215],[0.0955279,0.417875,0.158392,0.291707,0.0713435,0.717904,0.516147,0.197463,0.036434,0.123658,0.242759,0.352407,0.0521594],[0.0955621,0.418023,0.158383,0.291676, 0.07143, 0.71474,0.515909,0.197457,0.0364202, 0.12375, 0.24284,0.352408,0.052215],[0.0955906,  0.4175,0.158429,0.291691,0.0713995, 0.71486,0.516422,0.197513,0.0364215,0.123711,0.242866,0.352173,0.052195],[0.0956579,0.417656,0.158399,0.291655,0.0713695,0.714295,0.516124,0.197406,0.036423,0.123703,0.242862,0.352253,0.0521535]]

Mflops_blas = \
[[ 1.10728, 4.84544,  1.8259, 3.40663, 0.83193, 6.74192, 5.72991, 2.28081,0.427438, 1.43717, 2.80304, 4.07677,0.606932],[ 1.10656, 4.84579, 1.82584,  3.4066,0.832782, 6.74137, 5.73075, 2.28074,0.427361, 1.43663, 2.80365, 4.07852,0.606998],[ 1.10673, 4.84494, 1.82544, 3.40676,0.831916, 6.74501, 5.72797, 2.28028,0.427369,  1.4367, 2.80309, 4.07722,0.607057],[ 1.10686, 4.84542, 1.82568, 3.40746,0.832032,  6.7414, 5.72935,  2.2807, 0.42756, 1.43692,   2.803, 4.07693,0.607077],[ 1.10671, 4.84636, 1.82594, 3.40721,0.832111, 6.74235, 5.72915,  2.2802,0.427392, 1.43705, 2.80342, 4.07805,0.607169],[ 1.10683, 4.84485, 1.82576, 3.40688,0.832851, 6.74231, 5.72997, 2.28078,0.427704, 1.43694,  2.8031, 4.07651,0.607052],[ 1.10723, 4.84541, 1.82591, 3.40728,0.832167, 6.74169, 5.72925, 2.28041,0.427288, 1.43699, 2.80368, 4.07696,0.607055],[  1.1067, 4.84564, 1.82562, 3.40663,0.832016, 6.74278, 5.72829, 2.28017,0.427433, 1.43679, 2.80306, 4.07726,0.607311],[ 1.10675, 4.84714, 1.82626, 3.40712,0.831925, 6.74131, 5.73034,  2.2806,0.427401, 1.43733, 2.80277, 4.07654,0.606924],[ 1.10693, 4.84589, 1.82618, 3.40718,0.833385, 6.74208, 5.72856, 2.28035,0.427478, 1.43681,  2.8034, 4.07787,0.607086],[ 1.10751, 4.84463, 1.82569, 3.40689,0.831995, 6.74254, 5.73062, 2.28077,0.427452, 1.43691,  2.8034, 4.07661,0.606892],[ 1.10687, 4.84605, 1.82611, 3.40702,0.831818, 6.74179,   5.729, 2.28018,0.427479, 1.43679, 2.80336, 4.07676,0.606992],[ 1.10724, 4.84523, 1.82549, 3.40706, 0.83186, 6.74435, 5.72781, 2.28024, 0.42736, 1.43731, 2.80281, 4.07694,0.607069],[ 1.10678, 4.84682, 1.82613, 3.40703,0.833142, 6.74255, 5.73007, 2.28136,  0.4275, 1.43684, 2.80271, 4.07642,0.606977],[ 1.10739, 4.84566, 1.82536, 3.40725,0.831993, 6.74182, 5.72815, 2.28012,0.427359, 1.43705, 2.80379, 4.07829,  0.6073],[ 1.10662, 4.84461, 1.82532, 3.40674,0.832076, 6.74375, 5.73051, 2.28045, 0.42747, 1.43701, 2.80268, 4.07652,0.607122],[ 1.10689,   4.846, 1.82555,  3.4073,0.831963, 6.74228, 5.72974, 2.28033,0.427505, 1.43772, 2.80428, 4.07661,0.607153],[ 1.10662, 4.84522, 1.82589, 3.40704, 0.83304, 6.74366, 5.72859, 2.28024,0.427688, 1.43653, 2.80271,  4.0769,0.607039],[ 1.10654, 4.84581, 1.82564, 3.40725,0.832031, 6.74182, 5.72926,   2.281,0.427357, 1.43679, 2.80314,  4.0766,0.607607],[ 1.10662, 4.84627, 1.82559, 3.40685,0.833462, 6.74186, 5.73024, 2.28008,0.427517, 1.43692, 2.80363, 4.07816,  0.6071]]